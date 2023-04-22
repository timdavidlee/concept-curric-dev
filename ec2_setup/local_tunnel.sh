set -e

if [ $# -ne 1 ]; then
    echo "Usage $(basename $0) <target-host>"
    exit 1
fi


REMOTE_HOST=$1
# because aws ami, if debian / centos, the user will be different
USER='ec2-user'

ssh $USER@$REMOTE_HOST -N -L 8890:localhost:8890