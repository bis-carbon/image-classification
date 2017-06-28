"""
Create embedding to visualize tensors on tensorboard

  # classify_image_graph_def.pb:
  #   Binary representation of the GraphDef protocol buffer.
  # imagenet_synset_to_human_label_map.txt:
  #   Map from synset ID to a human readable string.
  # imagenet_2012_challenge_label_map_proto.pbtxt:
  #   Text representation of a protocol buffer mapping a label to synset ID.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import os, os.path
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


import argparse
import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
num_top_predictions = 5

DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz' # Link to download inception model
prediction_tensor = []
class NodeLookup(object):
  """Converts integer node ID's to human readable labels."""

  def __init__(self,model_DIR=None,
               label_lookup_path=None,
               uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.join(model_DIR, 'imagenet_2012_challenge_label_map_proto.pbtxt') # join model directory with the name of .pbtxt file
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(model_DIR, 'imagenet_synset_to_human_label_map.txt') # join model directory with the name of .txt file
    self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

  def load(self, label_lookup_path, uid_lookup_path):
    """Loads a human readable English name for each softmax node.

    Args:
      label_lookup_path: string UID to integer node ID.
      uid_lookup_path: string UID to human-readable string.

    Returns:
      dict from integer node ID to human-readable string.
    """
    if not tf.gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not tf.gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string

    # Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]

    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name

    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]



def create_graph(model_path):
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.

  with tf.gfile.FastGFile(os.path.join(model_path, 'classify_image_graph_def.pb'), 'rb') as f: # join model directory with the name of the graph .pb
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

def write_labels(i, label_name,LOG_DIR):
    metadata_file = open(os.path.join(LOG_DIR, 'metadata.tsv'), 'w')
    metadata_file.write('Name\tClass\n')
    metadata_file.write('%06d\t%s\n' % (i, label_name))
    metadata_file.close()

def accumulate_tensor(pred_tens):
    """appends tensor of pool_3 layer for each image """
    prediction_tensor.append(pred_tens.tolist())

def create_embedding(LOG_DIR):
    """create embedding and saves it to model.ckpt"""

    embedding_var = tf.Variable(prediction_tensor, name='inception')

    with tf.Session() as sess:
        # The embedding variable, which needs to be stored
        sess.run(embedding_var.initializer)
        summary_writer = tf.summary.FileWriter(LOG_DIR)
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name

        #attach metadata to embedding
        #Comment out if you dont have metadata
        embedding.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv') # join log directory with the name of the meta data

        # attach sprite image to embedding
        # Comment out if you don't have sprite image
        embedding.sprite.image_path = os.path.join(LOG_DIR, 'sprite.png') # join log directory with the name of the sprite image
        embedding.sprite.single_image_dim.extend([263, 320])

        projector.visualize_embeddings(summary_writer, config)
        saver = tf.train.Saver([embedding_var])
        saver.save(sess, os.path.join(LOG_DIR, 'model.ckpt'), 1) # out put embedding check point


def run_inference_on_image(image, model_DIR):
  """Runs inference on an image.

  Args:
    image: Image file name.

  Returns:
    prediction and html
  """
  if not tf.gfile.Exists(image):
    tf.logging.fatal('File does not exist %s', image)
  image_data = tf.gfile.FastGFile(image, 'rb').read()
  print (image)
  html = ''
  html += '<img src="%s" height="400"><br>' % (image)

  with tf.Session() as sess:
    # Some useful tensors:
    # 'softmax:0': A tensor containing the normalized prediction across
    #   1000 labels.
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    # Runs the softmax tensor by feeding the image_data as input to the graph.
    
    #softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    softmax_tensor = sess.graph.get_tensor_by_name('pool_3:0')
    predictions = sess.run(softmax_tensor,
                           {'DecodeJpeg/contents:0': image_data})
    predictions = np.squeeze(predictions)

    # Creates node ID --> English string lookup.
    node_lookup = NodeLookup(model_DIR=model_DIR)
    top_k = predictions.argsort()[-num_top_predictions:][::-1]
    accumulate_tensor(predictions)
    for node_id in top_k:
      human_string = node_lookup.id_to_string(node_id)
      score = predictions[node_id]
      html += '%s <br><br>' % (human_string + ": Score :" + str(score))  # added
      print('%s (score = %.5f)' % (human_string, score))
    answer = node_lookup.id_to_string(top_k[0])
    print ("This image is : ",answer)

  return answer, html


def maybe_download_and_extract():
  """Download and extract model tar file."""
  dest_directory = 'model_dir'
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def loop_on_images(tsv_file,DATA_DIR, model_DIR,log_DIR):
    #call run_inference_on_image() for each image in a given folder
    score = ''
    j = 0
    valid_images = [".jpg"]
    for f in os.listdir(DATA_DIR):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        print (f)
        label_name, html = run_inference_on_image(os.path.join(DATA_DIR, f), model_DIR)
        j += 1
        score +=html
        tsv_file.write('%06d\t%s\n' % (j, label_name))
    html_file = os.path.join(log_DIR, 'result_inception_pool3.html')
    open(html_file, 'w').write(score)

def _embedding(model_DIR, data_DIR, log_DIR):


  metadata_file = open(os.path.join(log_DIR, 'metadata.tsv'), 'w')
  metadata_file.write('Name\tClass\n')

  # Creates graph from saved GraphDef.
  create_graph(model_DIR)

  loop_on_images(metadata_file, data_DIR,model_DIR,log_DIR)
  metadata_file.close()
  create_embedding(log_DIR)

