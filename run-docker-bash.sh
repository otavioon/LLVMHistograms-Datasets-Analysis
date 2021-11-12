docker run -it --rm -v $PWD:$PWD -w $PWD -u $(id -u ${USER}):$(id -g ${USER}) devashishupadhyay/scikit-learn-docker /bin/bash
