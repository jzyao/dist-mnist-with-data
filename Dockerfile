FROM tf-dist-mnist-test:1.0
RUN python /var/tf_dist_mnist/dist_mnist.py && \
    tar -zcvf mnist-data.tgz /tmp/mnist-data/

FROM tf-dist-mnist-test:1.0
COPY --from=0 ADD mnist-data.tgz /tmp/
