import training
import training_parameters as tparams
import keras.optimizers
import image_database as imdb
import semantic_network_models as sem_models
import custom_losses
import custom_metrics
import dataset_construction
from keras.utils import to_categorical
import augmentation as aug

keras.backend.set_image_dim_ordering('tf')

INPUT_CHANNELS = 1
DATASET_NAME = "mydata"     # can choose a name if desired

# images numpy array should be of the shape: (number of images, image width, image height, 1)
# segs numpy array should be of the shape: (number of images, number of boundaries, image width)


# fill in this function to load your data for the training set with format/shape given above
def load_training_data():
    # FILL IN THIS FUNCTION TO LOAD YOUR DATA
    return #images, segs

# fill in this function to load your data for the validation set with format/shape given above
def load_validation_data():
    # FILL IN THIS FUNCTION TO LOAD YOUR DATA
    return  # images, segs

train_images, train_segs = load_training_data()
val_images, val_segs = load_validation_data()

train_labels = dataset_construction.create_all_area_masks(train_images, train_segs)
val_labels = dataset_construction.create_all_area_masks(val_images, val_segs)

NUM_CLASSES = train_segs.shape[1] + 1

train_labels = to_categorical(train_labels, NUM_CLASSES)
val_labels = to_categorical(val_labels, NUM_CLASSES)

train_imdb = imdb.ImageDatabase(images=train_images, labels=train_labels, name=DATASET_NAME, filename=DATASET_NAME, mode_type='fullsize', num_classes=NUM_CLASSES)
val_imdb = imdb.ImageDatabase(images=val_images, labels=val_labels, name=DATASET_NAME, filename=DATASET_NAME, mode_type='fullsize', num_classes=NUM_CLASSES)

# Stargardt models
model_residual_scSE = sem_models.resnet(8, 4, 2, 1, (3, 3), (2, 2), INPUT_CHANNELS, NUM_CLASSES, se='scSE')
model_residual_5deep = sem_models.resnet(8, 5, 2, 1, (3, 3), (2, 2), INPUT_CHANNELS, NUM_CLASSES)

opt_con = keras.optimizers.Adam
opt_params = {}     # default params
loss = custom_losses.dice_loss
metric = custom_metrics.dice_coef
epochs = 100
batch_size = 3

# augmentations
aug_fn_args_stargardt = [(aug.no_aug, {}), (aug.gaussian_noise_aug, {'variance': 'random', 'min': 250, 'max': 1000}), (aug.flip_aug, {'flip_type': 'left-right'}),
               (aug.combo_aug, [(aug.gaussian_noise_aug, {'variance': 'random', 'min': 250, 'max': 1000}), (aug.flip_aug, {'flip_type': 'left-right'})])]
aug_mode_stargardt = 'one'
aug_probs_stargardt = (0.25, 0.25, 0.25, 0.25)
aug_val_stargardt = False
aug_fly_stargardt = True

# no augmentations
aug_fn_args_stargardt_na = [(aug.no_aug, {})]
aug_mode_stargardt_na = 'none'
aug_probs_stargardt_na = []
aug_val_stargardt_na = False
aug_fly_stargardt_na = False


train_params = tparams.TrainingParams(model_residual_scSE, opt_con, opt_params, loss, metric, epochs, batch_size, model_save_best=True, aug_fn_args=aug_fn_args_stargardt, aug_mode=aug_mode_stargardt,
                                      aug_probs=aug_probs_stargardt, aug_val=aug_val_stargardt, aug_fly=aug_fly_stargardt)

training.train_network(train_imdb, val_imdb, train_params)
