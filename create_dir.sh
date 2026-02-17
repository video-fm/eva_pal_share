TASK="AuroraQian"
MODEL="pi05_droid"
DIR="results_shared" 
# DATE="time_denoise_0.4"
# DATE="$(date +%Y-%m-%d)"

mkdir -p $DIR/
mkdir -p $DIR/$TASK/ 
mkdir -p $DIR/$TASK/$MODEL    
mkdir -p $DIR/$TASK/$MODEL/success/
# mkdir -p $DIR/$TASK/$MODEL/success/$DATE
mkdir -p $DIR/$TASK/$MODEL/failure/
# mkdir -p $DIR/$TASK/$MODEL/failure/$DATE