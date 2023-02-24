#!/bin/bash
ext=`date +"%y_%m.tgz"`
root=.
mandatory="$root/README.md"
common="$root/mtc $root/mpc $root/set-up-env.sh $root/bin $root/examples $root/Dockerfile $root/modulus2207-patch $root/MTC_VERSION $root/build_docker_image.sh $root/make_docker_image_tarball.sh"

excl="--exclude=*/.* --exclude=*/.* --exclude=*/venv/* --exclude=*/__pycache__/* --exclude=*/outputs/* --exclude=*/venv-mtc-lab --exclude=*/.ipynb_checkpoints --exclude=*/*.pptx"

tar cvfz "Nvidia_Modulus_MTC_v"$ext $excl $mandatory $common 
