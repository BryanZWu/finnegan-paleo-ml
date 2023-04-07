"""forams dataset."""

from pathlib import Path
import tensorflow_datasets as tfds
import tensorflow as tf
import pandas as pd
import re
import os

_DESCRIPTION = """
The benthic forams dataset contains 10829 images resized to 224 x 224,
including 8024 benthic foraminifera. The two-label dataset provides one
label for the species (ordinally encoded from 0-54, with the encoding 0
reserved for non-forams), and a second boolean label for the fragmentation
state of the shell (1 for broken, and 0 for intact). For all images of 
debris and non-forams, the species label is 0 and the fragmentation label
is also 0. The dataset is provided as an 80-10-10 train-val-test split.
"""

_CITATION = """
@MISC{Kahanamoku2022-aw,
  title     = "Twenty-two thousand Common Era benthic foraminifera from the
               Santa Barbara Basin",
  author    = "Kahanamoku, Sara and Samuels-Fair, Maya and Kamel, Sarah M and
               Stewart, Da'shaun and Kahn, Leah and Titcomb, Max and Mei,
               Yingyan Alyssa and Bridge, R Cheyenne and Li, Yuerong Sophie and
               Sinco, Carolina and Epino, J T and Gonzalez-Marin, Gerson and
               Latt, Chloe and Fergus, Heather and Finnegan, Seth",
  abstract  = "Here we provide an image and 2D shape dataset of recent benthic
               foraminifera from two core records sampled from the center of
               the Santa Barbara Basin that span an ~800-year-long interval
               during the Common Era (1249-2008 CE). Information on more than
               36,000 objects is included, of which more than 22,000 are
               complete or partially damaged benthic foraminifera. Skeletonized
               objects classified also include planktonic foraminifera,
               ostracods, pteropods, diatoms, radiolarians, fish teeth and
               skeletal structures, shark dermal denticles, and benthic
               foraminifer test fragments. The image dataset was produced using
               a high-throughput imaging method (AutoMorph) designed to extract
               2D data from photographic images of fossils. This repository
               contains 8 distinct data types uploaded as distinct files, and
               includes the following: bulk\_images.zip: Bulk images with
               objects identified by segment boxed in red
               individual\_images.zip: EDF images of individual objects within
               the dataset identification\_files.zip: Classifications for
               individual objects, including both general categories and
               species-specific classifications (when possible) for benthic
               foraminifera cleaning\_scripts.zip: Directory containing R
               scripts used to clean object category misspellings or
               inconsistencies outline\_images.zip: EDF images of objects
               successfully extracted for 2D outlines and measurements;
               included for quality control. This includes one text file
               (unextracted\_objects\_2D.txt) listing objects with failed
               extractions 2d\_coordinates.zip: CSV files containing all
               extracted outline coordinates for each of the samples imaged, a
               text file of failed 2D extractions
               (unextracted\_objects\_2D.txt), and a summary CSV file including
               coordinates for all extracted objects (all\_coordinates.csv)
               2d\_properties.zip: 2D measurements for all objects
               metadata\_tables.zip: Tables 1, 2, and 3 and Supplementary Table
               1 from a forthcoming data descriptor publication, describing
               sample metadata, including site coordinates, sample names,
               object information, and summary statistics The dataset we
               provide here comprises the most extensive publicly available
               archive of benthic foraminiferal morphology and 2D morphological
               variation to date.",
  publisher = "Zenodo",
  year      =  2022
}
"""

_SPECIES_CLASSES = ['NOT FORAM', 'suggrunda eckisi', 'bulimina exilis',
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

_IMAGE_SIZE = 224
class Forams(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for forams dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }
  
  def __init__(self, image_label_dir, **kwargs):
    self.image_label_dir = image_label_dir
    super().__init__(**kwargs)


  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=(_IMAGE_SIZE, _IMAGE_SIZE, 3)),
            'species': tfds.features.ClassLabel(names=_SPECIES_CLASSES),
            'chamber_broken': tfds.features.ClassLabel(names=['unbroken', 'broken']),
        }),
        supervised_keys=('image', 'species'),
        homepage='https://doi.org/10.5281/zenodo.7274658',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    path = Path(self.image_label_dir)
    return {
        'train': self._generate_examples(path / 'train'),
        'val': self._generate_examples(path / 'val'), 
        'test': self._generate_examples(path / 'test'),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    labels_df  = pd.read_csv(Path(self.image_label_dir)/ 'image-labels.csv')
    labels_df = labels_df.set_index(['sample_name', 'object_num'])
    labels_df = labels_df.dropna(how='any')
    # drop any invalid rows, where the label isn't in the species list
    labels_df = labels_df[labels_df['species'].isin(_SPECIES_CLASSES)]
    labels_df = labels_df[labels_df['Broken'].isin(['unbroken', 'broken'])]

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
