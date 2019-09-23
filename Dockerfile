FROM tensorflow/tensorflow:1.5.0
ADD . /var/tf_dist_mnist && \
    python /var/tf_dist_mnist/dist_mnist.py

FROM tensorflow/tensorflow:1.5.0
COPY --from=0 /tmp/mnist-data/ /tmp/
ADD . /var/tf_dist_mnist
ENTRYPOINT ["python", "/var/tf_dist_mnist/dist_mnist.py"]
