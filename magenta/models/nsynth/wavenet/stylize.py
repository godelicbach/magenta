import pdb
import numpy as np
import tensorflow as tf
from scipy.io import wavfile

from magenta.models.nsynth import utils
from magenta.models.nsynth.wavenet.h512_bo16 import Config
from magenta.models.nsynth.wavenet import fastgen

FLAGS = tf.app.flags.FLAGS

# Load wavfile
def load_wav(file_path):
    return utils.load_audio(file_path)

def save_wav(data,file_path):
    wavfile.write(file_path,16000,data[:,:,0].T)

def load_style_nsynth(initial):
    """Load the NSynth autoencoder network for stylizing."""

    config = Config()
    with tf.device("/gpu:0"):
        initial = initial.reshape([1,-1,1]) # [Batch_size, length, channel]
        x = tf.Variable(initial) 
        graph = config.build({"wav": x}, is_training=False)
        graph.update({"X":x})

    return graph


# Calculate gram matrix from embedding.
def gram_matrix(embedding):
    embedding = tf.squeeze(embedding) # [1,125,16] -> [125,16]
    gram = tf.matmul(tf.transpose(embedding),embedding)

    return gram # 16x16 matrix.
    

# Calculate style loss by calculating squared error between two gram matrix.
def style_loss(base_embedding,style_embedding):
    
    base_embedding = tf.squeeze(base_embedding)
    style_embedding = tf.squeeze(style_embedding)
    
    

    base_gram = gram_matrix(base_embedding)
    style_gram = gram_matrix(style_embedding)

    embedding_size = tf.square(base_embedding.shape[0]*base_embedding.shape[1])
    embedding_size = tf.cast(embedding_size,tf.float32)

    return tf.nn.l2_loss(base_gram-style_gram)/embedding_size


# Update given wav file iteratively in order to modify its timbre.
def stylize(checkpoint_path,
            wav_path,save_path,style_embedding,iteration=1000,save_period=1):
    
    session_config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Graph().as_default(), tf.Session(config=session_config) as sess:
        # Load wav file.
        wav_data = load_wav(wav_path)
        # Load network and get base embedding.
        net = load_style_nsynth(wav_data)
        print "Model restored." 
        
        base_embedding = net["encoding"] 

        loss = style_loss(base_embedding,style_embedding) 

        opt = tf.train.AdamOptimizer()
        train_step = opt.minimize(loss,var_list=[net["X"]])


        # Initialize variables
        init_op = tf.global_variables_initializer()
        sess.run(init_op)


        for i in range(iteration):
            loss_value,_ = sess.run([loss,train_step]) 
            print loss_value,"  ",(i+1)

            if((i+1)%save_period==0):
              stylized_x = sess.run(net["X"])
              save_wav(stylized_x,
                       save_path+'keyboard_to_guitar_'+str(i+1)+'.wav')
              print "Saved %dth wavfile. (%d/%d)" % (i+1,i+1,iteration)


def main():
    wav_path = '/home/data/kyungsu/nsynth/keyboard_acoustic_004-042-075.wav'  
    style_embedding_path = '/home/data/kyungsu/nsynth/encoding_source/guitar_acoustic_014-042-100_embeddings.npy'
    checkpoint_path = '/home/data/kyungsu/nsynth/wavenet-ckpt/model.ckpt-200000' 
    save_path = '/home/data/kyungsu/nsynth/wav_result/gram_stylize/'
    style_embedding = np.load(style_embedding_path)
    stylize(checkpoint_path,wav_path,save_path,style_embedding)


if __name__ == '__main__':
    main()





