# -*- coding: utf-8 -*-
"""Methods called by API endpoints"""
from rest_framework.decorators import api_view
from django.http import HttpRequest, JsonResponse
from django.db import connection

import numpy as np
import pandas as pd
import scipy.spatial as sp
import fafbseg
import networkx as nx
import sqlite3

from cloudvolume import CloudVolume
from cloudvolume.datasource.precomputed.skeleton.sharded import ShardedPrecomputedSkeletonSource

from celery.task import task

from .settings import *

cv = CloudVolume(CLOUDVOLUME_URL_REMOTE, use_https=True, parallel=False)
cv.meta.info['skeletons'] = CLOUDVOLUME_SKELETONS
cv.skeleton.meta.refresh_info()
cv.skeleton = ShardedPrecomputedSkeletonSource(cv.skeleton.meta, cv.cache, cv.config)


def get_links(cursor, segment_id, where='segmentid_x', table='synlinks'):
    cols = ['offset', 'pre_x', 'pre_y', 'pre_z', 'post_x', 'post_y', 
            'post_z', 'scores', 'cleft_scores', 'segmentid_x', 'segmentid_y']
    cursor.execute('SELECT * from {} where {} = {};'.format(table, where, segment_id))
    pre_links = cursor.fetchall()
    return pd.DataFrame.from_records(pre_links, columns=cols)


def load_subgraph(cursor, start_segment_id, order = 0):
    """ Return a NetworkX graph with segments as nodes and synaptic connection
    as edges with synapse counts
    
    start_segment_id: starting segment for subgraph loading
    
    order: number of times to expand along edges
    """
    fetch_segments = set([start_segment_id])
    fetched_segments = set()
    g = nx.DiGraph()
        
    for ordern in range(order+1):
        if DEBUG: print('order', ordern, 'need to fetch', len(fetch_segments), 'segments')
        for i, segment_id in enumerate(list(fetch_segments)):
            
            if DEBUG: print('process segment', i, 'with segment_id', segment_id)

            if DEBUG: print('retrieve pre_links')
            pre_links = get_links(cursor, segment_id, where='segmentid_x')

            if DEBUG: print('retrieve post_links')
            post_links = get_links(cursor, segment_id, where='segmentid_y')

            if DEBUG: print('build graph ...')

            if DEBUG: print('number of pre_links', len(pre_links))
            for idx, r in pre_links.iterrows():
                from_id = int(r['segmentid_x'])
                to_id = int(r['segmentid_y'])
                if g.has_edge(from_id,to_id):
                    ed = g.get_edge_data(from_id,to_id)
                    ed['count'] += 1
                else:
                    g.add_edge(from_id, to_id, count= 1)

            if DEBUG: print('number of post_links', len(post_links))
            for idx, r in post_links.iterrows():
                from_id = int(r['segmentid_x'])
                to_id = int(r['segmentid_y'])
                if g.has_edge(from_id,to_id):
                    ed = g.get_edge_data(from_id,to_id)
                    ed['count'] += 1
                else:
                    g.add_edge(from_id, to_id, count= 1)

            fetched_segments.add(segment_id)
            
            if len(pre_links) > 0:
                all_postsynaptic_segments = set(pre_links['segmentid_y'])
                fetch_segments = fetch_segments.union(all_postsynaptic_segments)
            
            if len(post_links) > 0:
                all_presynaptic_segments = set(post_links['segmentid_x'])
                fetch_segments = fetch_segments.union(all_presynaptic_segments)
        
        # remove all segments that were already fetched
        fetch_segments = fetch_segments.difference(fetched_segments)
        
        # always remove 0
        if 0 in fetch_segments:
            fetch_segments.remove(0)

    return g

def get_presynaptic_skeletons(g, segment_id, synaptic_count_threshold = 0):
    res = set()
    for nid in g.predecessors(segment_id):
        if nid == segment_id or nid == 0:
            continue
        ed = g.get_edge_data(nid, segment_id)
        if ed['count'] > synaptic_count_threshold:
            res.add(nid)
    return list(res)


def get_postsynaptic_skeletons(g, segment_id, synaptic_count_threshold = 0):
    res = set()
    for nid in g.successors(segment_id):
        if nid == segment_id or nid == 0:
            continue
        ed = g.get_edge_data(segment_id, nid)
        if ed['count'] > synaptic_count_threshold:
            res.add(nid)
    return list(res)


@api_view(['GET'])
def get_neighbors_graph(request, segment_id):    
    conn = sqlite3.connect(SQLITE3_DB_PATH)
    cur = conn.cursor()
    g = load_subgraph(cur, segment_id, order = 0)
    from networkx.readwrite import json_graph
    return JsonResponse({'graph': json_graph.node_link_data(g)})


@api_view(['GET'])
def get_synapses(request, segment_id):    
    conn = sqlite3.connect(SQLITE3_DB_PATH)
    cur = conn.cursor()

    if DEBUG: print('retrieve pre_links')
    pre_links = get_links(cursor, segment_id, where='segmentid_x')

    if DEBUG: print('retrieve post_links')
    post_links = get_links(cursor, segment_id, where='segmentid_y')

    return JsonResponse({'pre_links': pre_links.to_json(), 'post_links': post_links.to_json()})


@api_view(['GET'])
def is_installed(request, project_id=None):
    """Check whether the extension circuitmap is installed."""
    return JsonResponse({'is_installed': True, 'msg': 'circuitmap is installed'})

@api_view(['GET'])
def index(request, project_id=None):
    return JsonResponse({'version': '0.1', 'app': 'circuitmap'})

@api_view(['GET'])
def test(request, project_id=None):
    task = testtask.delay()
    return JsonResponse({'msg': 'testtask'})

@task()
def testtask():
    print('testtask')


@api_view(['POST'])
def fetch_synapses(request: HttpRequest, project_id=None):
    x = int(round(float(request.POST.get('x', -1))))
    y = int(round(float(request.POST.get('y', -1))))
    z = int(round(float(request.POST.get('z', -1))))

    fetch_upstream = bool(request.POST.get('fetch_upstream', False ))
    fetch_downstream = bool(request.POST.get('fetch_downstream', False ))
    distance_threshold = int(request.POST.get('distance_threshold', 1000 ))
    active_skeleton_id = int(request.POST.get('active_skeleton', -1 ))
    upstream_syn_count = int(request.POST.get('upstream_syn_count', 5 ))
    downstream_syn_count = int(request.POST.get('downstream_syn_count', 5 ))

    pid = int(project_id)

    if x == -1 or y == -1 or z == -1:
        return JsonResponse({'project_id': pid, 'msg': 'Invalid location coordinates'})

    if active_skeleton_id == -1:

        # look up segment id at location and fetch synapses        
        try:
            segment_id = int(cv[x//2,y//2,z,0][0][0][0][0])
        except Exception as ex:
            if DEBUG: print('Exception occurred: {}'.format(ex))
            segment_id = None
            return JsonResponse({'project_id': pid, 'msg': 'No segment found at this location.', 'ex': ex})

        if segment_id is None or segment_id == 0:
            return JsonResponse({'project_id': pid, 'msg': 'No segment found at this location.'})
        else:
            if DEBUG: print('spawn task: import_autoseg_skeleton_with_synapses')

            task = import_autoseg_skeleton_with_synapses.delay(pid, 
                segment_id)

            if DEBUG: print('call: import_upstream_downstream_partners')
            task  = import_upstream_downstream_partners.delay(segment_id, fetch_upstream, fetch_downstream,
                pid, upstream_syn_count, downstream_syn_count)

            return JsonResponse({'project_id': pid, 'segment_id': str(segment_id)})

    else:
        # fetch synapses for manual skeleton
        task = import_synapses_for_existing_skeleton.delay(pid, 
            distance_threshold, active_skeleton_id)
        
        return JsonResponse({'project_id': pid})

@task()
def import_upstream_downstream_partners(segment_id, fetch_upstream, fetch_downstream, 
	pid, upstream_syn_count, downstream_syn_count):
    try:
        print('task: import_upstream_downstream_partners start', segment_id)

        # get all partners partners
        conn = sqlite3.connect(SQLITE3_DB_PATH)
        cur = conn.cursor()

        if DEBUG: print('load subgraph')
        g = load_subgraph(cur, segment_id)

        if DEBUG: print('start fetching ...')
        if fetch_upstream:
            for partner_segment_id in get_presynaptic_skeletons(g, segment_id, synaptic_count_threshold = upstream_syn_count):
                if DEBUG: print('spawn task for presynaptic segment_id', partner_segment_id)
                task = import_autoseg_skeleton_with_synapses.delay(pid, 
                    partner_segment_id)

        if fetch_downstream:
            for partner_segment_id in get_postsynaptic_skeletons(g, segment_id, synaptic_count_threshold = downstream_syn_count):
                if DEBUG: print('spawn task for postsynaptic segment_id', partner_segment_id)
                task = import_autoseg_skeleton_with_synapses.delay(pid, 
                    partner_segment_id)

    except Exception as ex:
        print('exception import_upstream_downstream_partners: ', ex)


@task()
def import_synapses_for_existing_skeleton(project_id, distance_threshold, active_skeleton_id,
    autoseg_segment_id = None):
    
    if DEBUG: print('task: import_synapses_for_existing_skeleton started')

    try:
        # retrieve skeleton with all nodes directly from the database
        cursor = connection.cursor()
        cursor.execute('''
            SELECT t.id, t.parent_id, t.location_x, t.location_y, t.location_z
            FROM treenode t
            WHERE t.skeleton_id = %s AND t.project_id = %s
            ''', (int(active_skeleton_id), int(project_id)))

        # convert record to pandas data frame
        skeleton = pd.DataFrame.from_records(cursor.fetchall(), 
            columns=['id', 'parent_id', 'x', 'y', 'z'])

        if DEBUG: print('skeleton shape', len(skeleton))

        # accessing the most recent autoseg data
        fafbseg.use_google_storage(GOOGLE_SEGMENTATION_STORAGE)

        if not autoseg_segment_id is None:
            if DEBUG: print('active skeleton {} is derived from segment id {}'.format(active_skeleton_id, autoseg_segment_id))
            overlapping_segmentids = set([int(autoseg_segment_id)])
        else:
            # retrieve segment ids
            segment_ids = fafbseg.segmentation.get_seg_ids(skeleton[['x','y','z']])

            if DEBUG:
                print('found segment ids for skeleton: ', segment_ids)

            overlapping_segmentids = set()
            for seglist in segment_ids:
                for s in seglist:
                    overlapping_segmentids.add(s)
        
        # store skeleton in kdtree for fast distance computations
        tree = sp.KDTree( skeleton[['x', 'y', 'z']] )

        connectors = {}
        treenode_connector = {}
        all_pre_links = []
        all_post_links = []

        # TODO: use local SQlite DB for now. later ingest data into 
        # circuitmap managed table (i.e circuit_synlinks)
        conn = sqlite3.connect(SQLITE3_DB_PATH)
        cur = conn.cursor()

        # retrieve synaptic links for each autoseg skeleton
        for segment_id in list(overlapping_segmentids):
            if DEBUG: print('process segment: ', segment_id)
            all_pre_links.append(get_links(cur, segment_id, 'segmentid_x'))
            all_post_links.append(get_links(cur, segment_id, 'segmentid_y'))
            
        all_pre_links_concat = pd.concat(all_pre_links)
        all_post_links_concat = pd.concat(all_post_links)

        if DEBUG:
            print('total nr prelinks collected', len(all_pre_links_concat))
            print('total nr postlinks collected', len(all_post_links_concat))

        if len(all_pre_links_concat) > 0:
            if DEBUG: print('find closest distances to skeleton for pre')
            res = tree.query(all_pre_links_concat[['pre_x','pre_y', 'pre_z']])
            all_pre_links_concat['dist'] = res[0]
            all_pre_links_concat['skeleton_node_id_index'] = res[1]
            for idx, r in all_pre_links_concat.iterrows():
                # skip link if beyond distance threshold
                if distance_threshold >= 0 and r['dist'] > distance_threshold:
                    continue
                if r['segmentid_x'] == r['segmentid_y']:
                    # skip selflinks
                    continue
                connector_id = CONNECTORID_OFFSET + int(r['offset']) * 10 
                if not connector_id in connectors:
                    connectors[connector_id] = r.to_dict()
                # add treenode_connector link
                treenode_connector[ \
                    (int(skeleton.loc[r['skeleton_node_id_index']]['id']), \
                     connector_id)] = {'type': 'presynaptic_to'}

        if len(all_post_links_concat) > 0:
            if DEBUG: print('find closest distances to skeleton for post')
            res = tree.query(all_post_links_concat[['post_x','post_y', 'post_z']])
            all_post_links_concat['dist'] = res[0]
            all_post_links_concat['skeleton_node_id_index'] = res[1]
                
            for idx, r in all_post_links_concat.iterrows():
                # skip link if beyond distance threshold
                if distance_threshold >= 0 and r['dist'] > distance_threshold:
                    continue
                if r['segmentid_x'] == r['segmentid_y']:
                    # skip selflinks
                    continue
                connector_id = CONNECTORID_OFFSET + int(r['offset']) * 10 
                if not connector_id in connectors:
                    connectors[connector_id] = r.to_dict()
                # add treenode_connector link
                treenode_connector[ \
                    (int(skeleton.loc[r['skeleton_node_id_index']]['id']), \
                     connector_id)] = {'type': 'postsynaptic_to'}

        # insert into database
        if DEBUG: print('fetch relations')
        cursor.execute("SELECT id,relation_name from relation where project_id = {project_id};".format(project_id=project_id))
        res = cursor.fetchall()
        relations = dict([(v,u) for u,v in res])

        with_multi = True
        queries = []
        query = 'BEGIN;'
        if with_multi:
            queries.append(query)
        else:
            cursor.execute(query)

        # insert connectors
        if DEBUG: print('start inserting connectors')
        for connector_id, r in connectors.items():
            q = """
        INSERT INTO connector (id, user_id, editor_id, project_id, location_x, location_y, location_z)
                    VALUES ({},{},{},{},{},{},{}) ON CONFLICT (id) DO NOTHING;
                """.format(
                connector_id,
                DEFAULT_IMPORT_USER,
                DEFAULT_IMPORT_USER,
                project_id,
                int(r['pre_x']), 
                int(r['pre_y']),
                int(r['pre_z'])) 

            if with_multi:
                queries.append(q)
            else:
                cursor.execute(q)

        # insert links
        # TODO: optimize based on scores
        confidence_value = 5

        if DEBUG: print('start insert links')
        for idx, val in treenode_connector.items():
            skeleton_node_id, connector_id = idx
            q = """
                INSERT INTO treenode_connector (user_id, project_id, 
                treenode_id,
                connector_id,
                relation_id, 
                skeleton_id,
                confidence)
                            VALUES ({},{},{},{},{},{},{}) ON CONFLICT ON CONSTRAINT treenode_connector_project_id_uniq DO NOTHING;;
                """.format(
                DEFAULT_IMPORT_USER,
                project_id,
                skeleton_node_id,
                connector_id,
                relations[val['type']],
                active_skeleton_id,
                confidence_value)

            if with_multi:
                queries.append(q)
            else:
                cursor.execute(q)

        q = 'COMMIT;'
        if with_multi:
            queries.append(q)
        else:
            cursor.execute(q)

        if with_multi:
            if DEBUG: print('run multiquery')
            cursor.execute('\n'.join(queries))


        if DEBUG: print('task: import_synapses_for_existing_skeleton started: done')
    except Exception as ex:
        print('Exception occurred: {}'.format(ex))


@task
def import_autoseg_skeleton_with_synapses(project_id, segment_id):

    try:
        # ID handling: method globally unique ID
        def mapping_skel_nid(segment_id, nid, project_id):
            max_nodes = 100000 # max. number of nodes / autoseg skeleton allowed
            nr_projects = 10 # max number of projects / instance allowed
            return int( int(segment_id) * max_nodes * nr_projects + int(nid) * nr_projects + int(project_id) )

        if DEBUG: print('task: import_autoseg_skeleton_with_synapses started')
        cursor = connection.cursor()

        # check if skeleton of autoseg segment_id was previously imported
        if DEBUG: print('check if already imported')
        cursor.execute('SELECT id, skeleton_id FROM treenode WHERE project_id = {} and id = {}'.format( \
            int(project_id), mapping_skel_nid(segment_id, 0, int(project_id))))
        res = cursor.fetchone()
        if not res is None:
            node_id, skeleton_class_instance_id = res
            if DEBUG: print('autoseg skeleton was previously imported. skip reimport. (current skeletonid is {})'.format(skeleton_class_instance_id))
        else:        
            # fetch and insert autoseg skeleton at location
            if DEBUG: print('fetch skeleton for segment_id {}'.format(segment_id))
            
            cv = CloudVolume(CLOUDVOLUME_URL, use_https=False, parallel=False)
            cv.meta.info['skeletons'] = CLOUDVOLUME_SKELETONS
            cv.skeleton.meta.refresh_info()
            cv.skeleton = ShardedPrecomputedSkeletonSource(cv.skeleton.meta, cv.cache, cv.config)

            s1 = cv.skeleton.get(int(segment_id))
            nr_of_vertices = len(s1.vertices)

            if DEBUG: print('autoseg skeleton for {} has {} nodes'.format(segment_id, nr_of_vertices))
            
            if DEBUG: print('generate graph for skeleton')
            g=nx.Graph()
            attrs = []
            for idx in range(nr_of_vertices):
                x,y,z=map(int,s1.vertices[idx,:]) 
                r = s1.radius[idx]
                attrs.append((int(idx),{'x':x,'y':y,'z':z,'r':float(r) }))
            g.add_nodes_from(attrs)
            edgs = []
            for u,v in s1.edges:
                edgs.append((int(u), int(v)))
            g.add_edges_from(edgs)

            # TODO: check if it skeleton already imported
            # this check depends on the chosen implementation
            if DEBUG: print('check number of connected components')
            nr_components = nx.number_connected_components(g)
            if nr_components > 1:
                if DEBUG: print('more than one component in skeleton graph. use only largest component')
                graph = max(nx.connected_component_subgraphs(g), key=len)
            else:
                graph = g

            # do relabeling and choose root node
            g2 = nx.relabel_nodes(g, lambda x: mapping_skel_nid(segment_id, x, project_id))
            root_skeleton_id = mapping_skel_nid(segment_id, 0, project_id)
            new_tree = nx.bfs_tree(g2, root_skeleton_id)

            if DEBUG: print('fetch relations and classes')
            cursor.execute("SELECT id,relation_name from relation where project_id = {project_id};".format(project_id=project_id))
            res = cursor.fetchall()
            relations = dict([(v,u) for u,v in res])

            cursor.execute("SELECT id,class_name from class where project_id = {project_id};".format(project_id=project_id))
            res = cursor.fetchall()
            classes = dict([(v,u) for u,v in res])

            
            query = """
            INSERT INTO class_instance (user_id, project_id, class_id, name)
                        VALUES ({},{},{},'{}') RETURNING id;
            """.format(DEFAULT_IMPORT_USER, project_id, classes['neuron'] ,"neuron {}".format(segment_id))
            if DEBUG: print(query)
            cursor.execute(query)
            neuron_class_instance_id = cursor.fetchone()[0]
            if DEBUG: print('got neuron', neuron_class_instance_id)

            query = """
            INSERT INTO class_instance (user_id, project_id, class_id, name)
                        VALUES ({},{},{},'{}') RETURNING id;
            """.format(DEFAULT_IMPORT_USER, project_id, classes['skeleton'] ,"skeleton {}".format(segment_id))
            if DEBUG: print(query)
            cursor.execute(query)
            skeleton_class_instance_id = cursor.fetchone()[0]
            if DEBUG: print('got skeleton', skeleton_class_instance_id)

            query = """
            INSERT INTO class_instance_class_instance (user_id, project_id, class_instance_a, class_instance_b, relation_id)
                        VALUES ({},{},{},{},{}) RETURNING id;
            """.format(DEFAULT_IMPORT_USER, project_id, skeleton_class_instance_id, neuron_class_instance_id, relations['model_of'])
            if DEBUG: print(query)
            cursor.execute(query)
            cici_id = cursor.fetchone()[0]

            # insert treenodes

            queries = []
            with_multi = True
            q = 'BEGIN;'
            if with_multi:
                queries.append(q)
            else:
                cursor.execute(q)

            # insert root node
            parent_id = ""
            n = g2.node[root_skeleton_id]
            query = """INSERT INTO treenode (id, project_id, location_x, location_y, location_z, editor_id,
                        user_id, skeleton_id, radius) VALUES ({},{},{},{},{},{},{},{},{}) ON CONFLICT (id) DO NOTHING;
                """.format(
                 root_skeleton_id,
                 project_id,
                 n['x'],
                 n['y'],
                 n['z'],
                 DEFAULT_IMPORT_USER,
                 DEFAULT_IMPORT_USER,
                 skeleton_class_instance_id,
                 n['r'])
            if with_multi:
                queries.append(query)
            else:
                cursor.execute(query)

            # insert all chidren
            for parent_id, skeleton_node_id in new_tree.edges(data=False):
                n = g2.node[skeleton_node_id]
                query = """INSERT INTO treenode (id,project_id, location_x, location_y, location_z, editor_id,
                            user_id, skeleton_id, radius, parent_id) VALUES ({},{},{},{},{},{},{},{},{},{}) ON CONFLICT (id) DO NOTHING;
                    """.format(
                     skeleton_node_id,
                     project_id,
                     n['x'],
                     n['y'],
                     n['z'],
                     DEFAULT_IMPORT_USER,
                     DEFAULT_IMPORT_USER,
                     skeleton_class_instance_id,
                     n['r'],
                    parent_id)
                if with_multi:
                    queries.append(query)
                else:
                    cursor.execute(query)

            query = 'COMMIT;'
            if with_multi:
                queries.append(query)
            else:
                cursor.execute(query)

            if DEBUG: print('run multiquery')
            if with_multi:
                cursor.execute('\n'.join(queries))

        # call import_synapses_for_existing_skeleton with autoseg skeleton as seed
        if DEBUG: print('call task: import_synapses_for_existing_skeleton')

        import_synapses_for_existing_skeleton(project_id, 
            -1,  skeleton_class_instance_id, segment_id)

        if DEBUG: print('task: import_autoseg_skeleton_with_synapses done')

    except Exception as ex:
        print('Exception occurred: {}'.format(ex))
