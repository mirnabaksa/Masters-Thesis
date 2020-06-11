nvidia-smi | grep 'python3' | awk '{ print $3 }' | xargs -n1 kill -9
