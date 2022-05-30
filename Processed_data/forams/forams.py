"""forams dataset."""

import tensorflow_datasets as tfds

_DESCRIPTION = """
A dataset with an image of forams and labels which correspond to their species.
Contains # number of 412x412 images in RGB format. 
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
  IMAGE_SIZE = 412 #TODO
  SPECIES_LIST = []

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
            'chamber_broken': tfds.features.ClassLabel(names=['broken', 'unbroken']), # TODO classlabel might not be correct for this? Can't find bool though.
            'chamber_size': tfds.features.ClassLabel(names=['small', 'medium']), # a single int. Will likely require regression
            'chamber_count': tfds.features.Tensor, # a single int. Will likely require regression
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=None,
        homepage='https://dataset-homepage/', # TODO
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # Consider hosting and downloading the data, but also for now find to just have it in the dir I think?
    path = tfds.core.tfds_path()

    # TODO(forams): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(path / 'train'),
        'val': self._generate_examples(path / 'val'), 
        'test': self._generate_examples(path / 'test'),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    # TODO(forams): Yields (key, example) tuples from the dataset
    csv = ... # TODO read CSV here. Consider using pandas DF to index
    # csv = pd.read_csv(path / 'masterCSVname.csv')
    for dir in path.glob('*'):
      yield dir, { # name of directory is unique
          'image': dir / 'img.png',
          'label': 'yes', # TODO actually get it working first and then troubleshoot examples. 
          'image': None, 
          'species': None, 
          'chamber_broken': None, 
          'chamber_size': None, 
          'chamber_count': None,
      }
