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



@api_view(['GET'])
def is_installed(request, project_id=None):
    """Check whether the extension circuitmap is installed."""
    return JsonResponse({'is_installed': True, 'msg': 'circuitmap is installed'})


@api_view(['GET'])
def index(request, project_id=None):
    return JsonResponse({'version': '0.1', 'app': 'circuitmap'})


@task()
def testtask():
    print('testtask')
    cv = CloudVolume(CLOUDVOLUME_URL, use_https=False, parallel=False)
    cv.meta.info['skeletons'] = CLOUDVOLUME_SKELETONS
    cv.skeleton.meta.refresh_info()
    cv.skeleton = ShardedPrecomputedSkeletonSource(cv.skeleton.meta, cv.cache, cv.config)
    s1 = cv.skeleton.get(4643129657)
    print("nr vert", len(s1.vertices))


@task()
def import_synapses_manual_skeleton(project_id, fetch_upstream, fetch_downstream,
    distance_threshold, active_skeleton_id, xres, yres, zres, autoseg_segment_id = None):
    
    if DEBUG: print('task: import_synapses_manual_skeleton started')

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

        def get_links(segment_id, where='segmentid_x', table='synlinks'):
            cols = ['offset', 'pre_x', 'pre_y', 'pre_z', 'post_x', 'post_y', 
                    'post_z', 'scores', 'cleft_scores', 'segmentid_x', 'segmentid_y']
            cur.execute('SELECT * from {} where {} = {};'.format(table, where, segment_id))
            pre_links = cur.fetchall()
            return pd.DataFrame.from_records(pre_links, columns=cols)

        # retrieve synaptic links for each autoseg skeleton
        for segment_id in list(overlapping_segmentids):
            if DEBUG: print('process segment: ', segment_id)
            all_pre_links.append(get_links(segment_id, 'segmentid_x'))
            all_post_links.append(get_links(segment_id, 'segmentid_y'))
            
        all_pre_links_concat = pd.concat(all_pre_links)
        all_post_links_concat = pd.concat(all_post_links)

        if DEBUG:
            print('total nr prelinks collected', len(all_pre_links_concat))
            print('total nr postlinks collected', len(all_post_links_concat))

        if len(all_pre_links_concat) > 0:
            if DEBUG: print('find closest distances to skeleton for pre')
            res = tree.query(all_pre_links_concat[['pre_x','pre_y', 'pre_z']] * \
                np.array([xres,yres,zres]))
            all_pre_links_concat['dist'] = res[0]
            all_pre_links_concat['skeleton_node_id_index'] = res[1]
            for idx, r in all_pre_links_concat.iterrows():
                # skip link if beyond distance threshold
                if distance_threshold >= 0 and r['dist'] > distance_threshold:
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
            res = tree.query(all_post_links_concat[['post_x','post_y', 'post_z']] * \
                np.array([xres,yres,zres]))
            all_post_links_concat['dist'] = res[0]
            all_post_links_concat['skeleton_node_id_index'] = res[1]
                
            for idx, r in all_post_links_concat.iterrows():
                # skip link if beyond distance threshold
                if distance_threshold >= 0 and r['dist'] > distance_threshold:
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

        # insert connectors
        if DEBUG: print('start inserting connectors')
        for connector_id, r in connectors.items():
            q = """
        INSERT INTO connector (id, user_id, editor_id, project_id, location_x, location_y, location_z)
                    VALUES ({},{},{},{},{},{},{});
                """.format(
                connector_id,
                DEFAULT_IMPORT_USER,
                DEFAULT_IMPORT_USER,
                project_id,
                int(r['pre_x'] * xres), 
                int(r['pre_y'] * yres),
                int(r['pre_z'] * zres)) 
                
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
                            VALUES ({},{},{},{},{},{},{});
                """.format(
                DEFAULT_IMPORT_USER,
                project_id,
                skeleton_node_id,
                connector_id,
                relations[val['type']],
                active_skeleton_id,
                confidence_value)

            cursor.execute(q)

        if DEBUG: print('task: import_synapses_manual_skeleton started: done')
    except Exception as ex:
        print('Exception occurred: {}'.format(ex))


@api_view(['POST'])
def fetch_synapses(request: HttpRequest, project_id=None):

    # TODO: issue with coordinate are float
    x = int(round(float(request.POST.get('x', -1))))
    y = int(round(float(request.POST.get('y', -1))))
    z = int(round(float(request.POST.get('z', -1))))

    xres = int(request.POST.get('xres', 1))
    yres = int(request.POST.get('yres', 1))
    zres = int(request.POST.get('zres', 1))

    fetch_upstream = bool(request.POST.get('fetch_upstream', False ))
    fetch_downstream = bool(request.POST.get('fetch_downstream', False ))
    distance_threshold = int(request.POST.get('distance_threshold', 1000 ))
    active_skeleton_id = int(request.POST.get('active_skeleton', -1 ))

    pid = int(project_id)

    if x == -1 or y == -1 or z == -1:
        return JsonResponse({'project_id': pid, 'msg': 'Invalid location coordinates'})

    if active_skeleton_id == -1:
        # look up segment id at location and fetch synapses
        cv = CloudVolume(CLOUDVOLUME_URL_REMOTE, use_https=True, parallel=False)
        cv.meta.info['skeletons'] = CLOUDVOLUME_SKELETONS
        cv.skeleton.meta.refresh_info()
        cv.skeleton = ShardedPrecomputedSkeletonSource(cv.skeleton.meta, cv.cache, cv.config)
        try:
            segment_id = cv[x//2,y//2,z,0][0][0][0][0]
        except Exception as ex:
            print('Exception occurred: {}'.format(ex))
            segment_id = None
            return JsonResponse({'project_id': pid, 'msg': 'No segment found at this location.', 'ex': ex})

        if segment_id is None:
            return JsonResponse({'project_id': pid, 'msg': 'No segment found at this location.'})
        else:
            if DEBUG: print('before task delay')
            task = import_autoseg_skeleton_with_synapses.delay(pid, 
                fetch_upstream, fetch_downstream, 
                segment_id, xres, yres, zres)
            if DEBUG: print('afer task delay')
            return JsonResponse({'project_id': pid, 'segment_id': str(segment_id)})

    else:
        # fetch synapses for manual skeleton
        task = import_synapses_manual_skeleton.delay(pid, 
            fetch_upstream, fetch_downstream, distance_threshold, 
            active_skeleton_id,
            xres, yres, zres)
        return JsonResponse({'project_id': pid})

    
@task
def import_autoseg_skeleton_with_synapses(project_id, fetch_upstream, fetch_downstream, segment_id, xres, yres, zres):

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

            cursor.execute('BEGIN;')
            
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

            # insert root node
            parent_id = ""
            n = g2.node[root_skeleton_id]
            query = """INSERT INTO treenode (id, project_id, location_x, location_y, location_z, editor_id,
                        user_id, skeleton_id, radius) VALUES ({},{},{},{},{},{},{},{},{});
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
            if DEBUG: print(query)
            cursor.execute(query)

            # insert all chidren
            for parent_id, skeleton_node_id in new_tree.edges(data=False):
                n = g2.node[skeleton_node_id]
                query = """INSERT INTO treenode (id,project_id, location_x, location_y, location_z, editor_id,
                            user_id, skeleton_id, radius, parent_id) VALUES ({},{},{},{},{},{},{},{},{},{});
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
                if DEBUG: print(query)
                cursor.execute(query)

            cursor.execute('COMMIT;')

        # call import_synapses_manual_skeleton with autoseg skeleton as seed
        if DEBUG: print('call task: import_synapses_manual_skeleton')

        import_synapses_manual_skeleton.delay(project_id, 
            fetch_upstream, fetch_downstream, -1, 
            skeleton_class_instance_id,
            xres, yres, zres, segment_id)

        if DEBUG: print('task: import_autoseg_skeleton_with_synapses done')

    except Exception as ex:
        print('Exception occurred: {}'.format(ex))
