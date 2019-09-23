FROM tensorflow/tensorflow:1.5.0
ADD . /var/tf_dist_mnist
RUN python /var/tf_dist_mnist/download_data.py

FROM tensorflow/tensorflow:1.5.0
COPY --from=0 /tmp/mnist-data/ /tmp/
ADD . /var/tf_dist_mnist
ENTRYPOINT ["python", "/var/tf_dist_mnist/dist_mnist.py"]
