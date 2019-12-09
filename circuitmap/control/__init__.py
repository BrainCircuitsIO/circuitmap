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

from cloudvolume import CloudVolume
from cloudvolume.datasource.precomputed.skeleton.sharded import ShardedPrecomputedSkeletonSource

from celery.task import task

from .settings import *

cv = CloudVolume(CLOUDVOLUME_URL, use_https=True)
cv.meta.info['skeletons'] = CLOUDVOLUME_SKELETONS
cv.skeleton.meta.refresh_info()
cv.skeleton = ShardedPrecomputedSkeletonSource(cv.skeleton.meta, cv.cache, cv.config)


@api_view(['GET'])
def is_installed(request, project_id=None):
    """Check whether the extension circuitmap is installed."""
    return JsonResponse({'is_installed': True, 'msg': 'circuitmap is installed'})


@api_view(['GET'])
def index(request, project_id=None):
    return JsonResponse({'version': '0.1', 'app': 'circuitmap'})


@task
def import_autoseg_skeleton_with_synapses(project_id, fetch_upstream, fetch_downstream,
    segment_id, xres, yres, zres):
    
    if DEBUG: print('task: import_autoseg_skeleton_with_synapses started')
    cursor = connection.cursor()

    # fetch and insert autoseg skeleton at location
    if DEBUG: print('fetch skeleton for segment_id {}'.format(segment_id))
    
    s1 = cv.skeleton.get(segment_id)
    nr_of_vertices = len(s1.vertices)
    if DEBUG: print('autoseg skeleton for {} has {} nodes'.format(segment_id,
     nr_of_vertices))
    
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

    # TODO: depending on the implementation, choose a more sensible
    # root node
    root_skeleton_id = list(g.nodes())[0]
    new_tree = nx.bfs_tree(g, root_skeleton_id)

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
    n = g.node[root_skeleton_id]
    query = """INSERT INTO treenode (project_id, location_x, location_y, location_z, editor_id,
                user_id, skeleton_id, radius) VALUES ({},{},{},{},{},{},{},{});
        """.format(
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
        n = g.node[skeleton_node_id]
        query = """INSERT INTO treenode (project_id, location_x, location_y, location_z, editor_id,
                    user_id, skeleton_id, radius, parent_id) VALUES ({},{},{},{},{},{},{},{},{});
            """.format(
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

    # TODO: call import_synapses_manual_skeleton with autoseg skeleton as seed

    if DEBUG: print('task: import_autoseg_skeleton_with_synapses done')


@task()
def import_synapses_manual_skeleton(project_id, fetch_upstream, fetch_downstream,
    distance_threshold, active_skeleton_id, xres, yres, zres):
    
    if DEBUG: print('task: import_synapses_manual_skeleton started')

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

    # accessing the most recent autoseg data
    fafbseg.use_google_storage(GOOGLE_SEGMENTATION_STORAGE)

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
        cols = ['ids', 'pre_x', 'pre_y', 'pre_z', 'post_x', 'post_y', 
                'post_z', 'scores', 'segmentid_x', 'segmentid_y', 'max', 'index']
        cur.execute('SELECT * from {} where {} = {};'.format(table, where, segment_id))
        pre_links = cur.fetchall()
        return pd.DataFrame.from_records(pre_links, columns=cols)

    # retrieve synaptic links for each autoseg skeleton
    for segment_id  in list(overlapping_segmentids):
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
            if r['dist'] > distance_threshold:
                continue
            connector_id = CONNECTORID_OFFSET + int(r['index']) * 10 
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
            if r['dist'] > distance_threshold:
                continue

            connector_id = CONNECTORID_OFFSET + int(r['index']) * 10 
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
            int( (r['pre_z']+1) * zres)) 
            # TODO: remove offset by 1 due to preprocessing when shifted in original data
            
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

    if x == -1 or y == -1 or z == -1:
        return JsonResponse({'project_id': project_id, 'msg': 'Invalid location coordinates'})

    if active_skeleton_id == -1:
        # look up segment id at location and fetch synapses
        try:
            segment_id = cv[x//2,y//2,z,0][0][0][0][0]
        except:
            segment_id = None

        task = import_autoseg_skeleton_with_synapses.delay(project_id, 
            fetch_upstream, fetch_downstream, 
            segment_id, xres, yres, zres)

        return JsonResponse({'project_id': project_id, 'segment_id': str(segment_id)})

    else:
        # fetch synapses for manual skeleton
        task = import_synapses_manual_skeleton.delay(project_id, 
            fetch_upstream, fetch_downstream, distance_threshold, 
            active_skeleton_id,
            xres, yres, zres)
        return JsonResponse({'project_id': project_id})

    
