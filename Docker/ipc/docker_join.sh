#!/usr/bin/env bash

BASH_OPTION=bash

IMG=argnctu/funie-gan:ipc

xhost +
containerid=$(docker ps -aqf "ancestor=${IMG}") && echo $containerid
docker exec -it \
    --privileged \
    -e DISPLAY=${DISPLAY} \
    -e LINES="$(tput lines)" \
    -e ROS_MASTER_URI=$ROS_MASTER_URI \
    -e ROS_IP=$ROS_IP \
    ${containerid} \
    $BASH_OPTION
xhost -