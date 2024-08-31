#!/bin/bash

docker run -it --device /dev/fuse --privileged duumviri_no_brave_source /bin/bash
#    --device /dev/fuse \
        #    --cap-add SYS_ADMIN \
        #    --security-opt apparmor:unconfined \