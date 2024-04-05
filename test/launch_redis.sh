CMD="docker run --rm -p 6379:6379 --name test_redis redis/redis-stack-server:latest redis-server /etc/redis-stack.conf --protected-mode no --bind 0.0.0.0 --loglevel debug"
echo "$ "$CMD
eval $CMD 2> /dev/null > /dev/null