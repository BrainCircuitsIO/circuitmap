# -*- coding: utf-8 -*-
"""Methods called by API endpoints"""
from rest_framework.decorators import api_view
from django.http import HttpRequest, JsonResponse

from cloudvolume import CloudVolume
from cloudvolume.datasource.precomputed.skeleton.sharded import ShardedPrecomputedSkeletonSource

CLOUDVOLUME_URL = 'precomputed://gs://fafb-ffn1-20190805/segmentation/'
CLOUDVOLUME_SKELETONS = 'skeletons_32nm_nothresh'

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
    return JsonResponse({'test': 'test message'})

@api_view(['POST'])
def segment_lookup(request: HttpRequest, project_id=None):
	x = int(request.POST.get('x', None))
	y = int(request.POST.get('y', None))
	z = int(request.POST.get('z', None))
	try:
		segment_id = cv[x//2,y//2,z,0][0][0][0][0]
	except:
		segment_id = None

	return JsonResponse({'project_id': project_id, 'segment_id': str(segment_id)})
