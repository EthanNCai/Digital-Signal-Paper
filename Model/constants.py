
import enum


@enum.unique
class Dataset(enum.Enum):

  LBS = "librispeech"

  BSD = "birdsong_detection"

  MUSAN = "musan"

  AS = "audioset"

  TUT = "tut_2018"

  SPCV1 = "speech_commands_v1"

  SPCV2 = "speech_commands"

  NSYNTH_INST = "nsynth_instrument_family"

  VOXCELEB = "voxceleb"

  VOXFORGE = "voxforge"

  CREMA_D = "crema_d"


@enum.unique
class TrainingMode(enum.Enum):

  SSL = "self_supervised"

  SUP = "supervised"

  RND = "random"

  DS = "downstream"


@enum.unique
class SimilarityMeasure(enum.Enum):


  DOT = "dot_product"

  BILINEAR = "bilinear_product"