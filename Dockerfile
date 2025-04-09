FROM public.ecr.aws/lambda/python:3.10

COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN python3 -m pip install -r ${LAMBDA_TASK_ROOT}/requirements.txt

COPY constants.py ${LAMBDA_TASK_ROOT}
COPY utils.py ${LAMBDA_TASK_ROOT}
COPY main.py ${LAMBDA_TASK_ROOT}

CMD ["main.handler"]
