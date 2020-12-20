###  Get Base Image - Linux alpine:3.7
FROM alpine:3.7

### Install Java (Got this directly on web)
RUN apk update \
&& apk upgrade \
&& apk add --no-cache bash \
&& apk add --no-cache --virtual=build-dependencies unzip \
&& apk add --no-cache curl \
&& apk add --no-cache openjdk8-jre


###  Install Python, PIP (Got this directly on web)

RUN apk add --no-cache python3 \
&& python3 -m ensurepip \
&& pip3 install --upgrade pip setuptools \
&& rm -r /usr/lib/python*/ensurepip && \
if [ ! -e /usr/bin/pip ]; then ln -s pip3 /usr/bin/pip ; fi && \
if [[ ! -e /usr/bin/python ]]; then ln -sf /usr/bin/python3 ; fi

### Removing cache files
RUN rm -r /root/.cache
#rm -r /root/.cache

RUN apk update && apk add bash

#### removed this command as easy_install will be deprecated in near future.
#RUN easy_install pip
#RUN pip install



##### upgrade pip
RUN pip install --upgrade pip

RUN apk add --update  python python3 python-dev python3-dev gfortran py-pip build-base
RUN apk update && apk add --no-cache libc6-compat
RUN BLAS=~/src/BLAS/libfblas.a LAPACK=~/src/lapack-3.5.0/liblapack.a pip install -v numpy==1.14

RUN  pip install wheel
RUN  pip install pyspark==2.3.2 --no-cache-dir
RUN  pip install findspark
RUN  pip install numpy

RUN pwd

### Copy the dataset to docker image
COPY postsubmissionTest.py postsubmissionTest.py
COPY TrainingDataset.csv TrainingDataset.csv
COPY rfModel rfModel

RUN ls -la /*

### 8. Entrypoint when docker run is called so that command line arguments could be passed to this file
ENTRYPOINT ["/traintest.py"]


##CMD ["python3", "traintest.py"]
	
