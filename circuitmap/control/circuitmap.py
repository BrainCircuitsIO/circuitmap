from celery.task import task

@task()
def testtask():
	print("Test task")