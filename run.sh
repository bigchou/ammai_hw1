python main.py --logfile triplet.txt --loss triplet --l2norm --dist l2 --epoch 50 --batch 128 --outdir triplet --num_triplets_train 150000 --training_triplets_path ""
python main.py --logfile sphereface.txt --loss sphereface --dist cosine --batch 256 --epoch 50 --outdir sphereface
python main.py --logfile softmax.txt --loss softmax --dist cosine --epoch 50 --batch 256 --outdir softmax
