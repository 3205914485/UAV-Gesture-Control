echo -n "Enter train name: "
read train_name

python -m global_selector >> "/data4/zst/uav/smart/results/$train_name.txt"