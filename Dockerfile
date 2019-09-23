FROM tensorflow/tensorflow:1.5.0
ADD . /var/tf_dist_mnist
ENTRYPOINT ["python", "/var/tf_dist_mnist/dist_mnist.py"]

FROM tensorflow/tensorflow:1.5.0
COPY --from=0 ADD mnist-data.tgz /tmp/
ADD . /var/tf_dist_mnist
ENTRYPOINT ["python", "/var/tf_dist_mnist/dist_mnist.py"]
