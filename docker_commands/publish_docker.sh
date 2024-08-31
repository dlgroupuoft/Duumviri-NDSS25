
# container id comes from ```docker ps```
docker commit 420de020705d duumviri_no_brave_source

docker tag duumviri_no_brave_source 8759s/duumviri_no_brave_source

docker push 8759s/duumviri_no_brave_source