extra_param=""
skip_count=2
for param in "$@"; do
  if [ $skip_count -gt 0 ]; then
    skip_count=`expr $skip_count - 1`;
  else
    extra_param=$extra_param" "$param;
  fi
done

command="nohup python3"$2" -m fl."$1""$extra_param

echo $command

killall python3 2> /dev/null
rm -rf nohup.out
$command &