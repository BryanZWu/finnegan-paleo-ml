"""forams dataset."""

from pathlib import Path
import tensorflow_datasets as tfds
import tensorflow as tf
import pandas as pd
import re
import os

_DESCRIPTION = """
A dataset with an image of forams and labels which correspond to their species.
Contains # number of 416x416 images in RGB format. 
"""

# TODO(forams): BibTeX citation
_CITATION = """
"""


class Forams(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for forams dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }
  IMAGE_SIZE = 416
  SPECIES_LIST = ['NOT FORAM', 'suggrunda eckisi', 'bulimina exilis',
    'nonionella stella', 'melonis affinis', 'bolivina argentea', 'fursenkoina bradyi',
    'bolivina seminuda', 'chilostomella oolina', 'bolivina sp. a', 'bolivina seminuda var. humilis',
    'bolivina spissa', 'cibicidoides sp. a', 'bolivina alata', 'cibicidoides wuellerstorfi',
    'chilostomella ovoidea', 'bolivina pacifica', 'nonionella decora', 'cassidulina crassa',
    'globocassidulina subglobosa', 'cassidulina minuta', 'epistominella exigua', 'oolina squamosa',
    'pyrgo murrhina', 'pullenia elegans', 'buccella peruviana', 'gyroidina subtenera',
    'bolivinita minuta', 'cassidulina carinata', 'alabaminella weddellensis', 'anomalinoides minimus',
    'uvigerina peregrina', 'pullenia bulloides', 'lenticulina sp. a', 'epistominella pulchella',
    'uvigerina interruptacostata', 'cassidulina auka', 'fursenkoina complanata', 'epistominella sp. a',
    'melonis pompilioides', 'laevidentalina sp. a', 'bolivina interjuncta', 'praeglobobulimina spinescens',
    'cassidulina delicata', 'globocassidulina neomargareta', 'triloculina trihedra', 'globobulimina barbata',
    'bolivina ordinaria', 'astrononion stellatum', 'epistominella obesa', 'epistominella pacifica',
    'fursenkoina pauciloculata', 'pyrgo sp. a', 'epistominella sandiegoensis', 'angulogerina angulosa']
  
  def __init__(self, image_label_dir, **kwargs):
    print('image_label_dir', image_label_dir)
    self.image_label_dir = image_label_dir
    super().__init__(**kwargs)


  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(forams): Specifies the tfds.core.DatasetInfo object. Expect something that looks like this: https://keras.io/guides/functional_api/#models-with-multiple-inputs-and-outputs
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=(self.IMAGE_SIZE, self.IMAGE_SIZE, 3)), # TODO confirm size
            'species': tfds.features.ClassLabel(names=self.SPECIES_LIST), # TODO confirm species list
            'chamber_broken': tfds.features.ClassLabel(names=['unbroken', 'broken']), # TODO classlabel might not be correct for this? Can't find bool though.
            # 'chamber_size': tfds.features.ClassLabel(names=['small', 'medium']), 
            # 'chamber_count': tfds.features.Tensor(shape=(), dtype=tf.dtypes.int16), # a single int. Will likely require regression
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'species'),
        homepage='https://dataset-homepage/', # TODO
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    path = Path(self.image_label_dir)
    return {
        'train': self._generate_examples(path / 'train'),
        'validation': self._generate_examples(path / 'val'), 
        'test': self._generate_examples(path / 'test'),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    labels_df  = pd.read_csv(Path(self.image_label_dir)/ 'image-labels.csv')
    labels_df = labels_df.set_index(['sample_name', 'object_num'])
    labels_df = labels_df.dropna(how='any')
    # drop any invalid rows, where the label isn't in the species list
    # labels_df = labels_df[labels_df['species'].isin(self.SPECIES_LIST)]
    # labels_df = labels_df[labels_df['chamber_broken'].isin(['unbroken', 'broken'])]

    # ACCESS via: labels_df.loc[('MV1012-BC-2', 1)]

    for image_path in path.glob('*.jpg'):
      image_name = image_path.name
      sample_name, object_number = re.match(r'(.+)_obj(\d+)', image_name).groups()
      object_number = int(object_number)
      yield image_name, { # name of image is unique
          'image': image_path, # image path converted automatically to tensor on load
          'species': labels_df.loc[(sample_name, object_number), 'species'],
          'chamber_broken': labels_df.loc[(sample_name, object_number), 'Broken'], 
          # 'species': None, 
          # 'chamber_size': None, 
          # 'chamber_count': None,
      }
