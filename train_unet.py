import numpy as np
import pandas as pd
import os
import cv2
import imageio
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, \
    Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.applications import MobileNetV2
import imgaug.augmenters as iaa
from sklearn.utils import shuffle
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import paths
import model_hyperparameters
import dataset_info
import image_config

from dataset_info import *
from paths import *
from image_config import *
from model_hyperparameters import *

def preprocess_image_mask(image, mask):
    """
    Preprocesses the input image and mask by applying data augmentation and normalization.

    Parameters:
        images (numpy.ndarray): The batch of input images.
        mask (numpy.ndarray): The batch of corresponding segmentation masks.

    Returns:
        tuple: A tuple containing the preprocessed images and masks.
    """

    # Data augmentation
    seq = iaa.Sequential( [
        iaa.Fliplr( 0.5 ),  # Apply horizontal flips with a probability of 0.5
        iaa.Flipud( 0.5 ),  # Apply vertical flips with a probability of 0.5
        iaa.Affine( rotate=(-10, 10) ),  # Apply random rotations between -10 and 10 degrees
        iaa.GaussianBlur( sigma=(0, 0.3) ),  # Apply Gaussian blur with a sigma between 0 and 0.5
    ] )

    mask = mask.astype(np.uint8)  # Convert to uint8 data type
    mask = np.expand_dims(mask, axis=-1)  # Add an extra dimension for C
    segmap = SegmentationMapsOnImage(mask, shape=image.shape)

    # Apply augmentation to the images and masks
    augmented = seq( image=image, segmentation_maps=segmap )
    images_augmented = augmented[0]  # Access the augmented images
    mask_augmented = augmented[1].get_arr()  # Access the augmented segmentation masks

    # Normalize pixel values to the range [0, 1]
    images_augmented = images_augmented / 255.0

    return images_augmented, mask_augmented


def data_generator(X, y, batch_size):
    """
    Generator function to yield batches of images and masks for training.

    Parameters:
        X (list): List of image filenames.
        y (list): List of encoded mask values.
        batch_size (int): Size of each batch.

    Yields:
        tuple: A tuple containing the batch of images and masks.
    """
    while True:
        # Iterate over the entire dataset in batches
        for i in range( 0, len( X ), batch_size ):
            batch_X = []
            batch_y = []
            # Iterate over the current batch
            for j in range( i, min( i + batch_size, len( X ) ) ):
                # Load and preprocess the image
                image_path = train_dir_img + X[j]
                image = cv2.imread( image_path )
                image = cv2.cvtColor( image, cv2.COLOR_BGR2RGB )  # Convert image to RGB format
                # Decode the encoded mask and preprocess it
                mask = decode_rle( y[j], (height, width) )
                image, mask = preprocess_image_mask( image, mask )
                batch_X.append( image )
                batch_y.append( mask )

            # Yield the batch of images and masks
            yield np.array( batch_X ), np.array( batch_y )

def decode_rle(encoded_pixels, shape):
    """
    Function to decode the RLE (Run-Length Encoding) encoded pixels into a binary mask image.

    Parameters:
        encoded_pixels (str): The RLE encoded pixel values.
        shape (tuple): The shape of the mask image (height, width).

    Returns:
        numpy.ndarray: The decoded binary mask image.

    """
    # Create an array to store the mask image
    mask_img = np.zeros( (shape[0] * shape[1], 1), dtype=np.float32 )

    # Check if the RLE encoding is not NaN (not a null value)
    if pd.notna( encoded_pixels ):
        # Split the encoded pixels into a list of integers
        rle = list( map( int, encoded_pixels.split( ' ' ) ) )
        pixel, pixel_count = [], []
        # Separate the pixel values and their counts into separate lists
        [pixel.append( rle[i] - 1 ) if i % 2 == 0 else pixel_count.append( rle[i] ) for i in range( 0, len( rle ) )]
        # Create a list of pixel ranges based on the pixel values and counts
        rle_pixels = [list( range( pixel[i], pixel[i] + pixel_count[i] ) ) for i in range( 0, len( pixel ) )]
        # Flatten the list of pixel ranges into a single list of pixel indices
        rle_mask_pixels = sum( rle_pixels, [] )

        # Try to set the corresponding pixels in the mask image to 1 based on the RLE indices
        try:
            mask_img[rle_mask_pixels] = 1.
        # Catch any potential IndexError (e.g., if the RLE indices exceed the mask image dimensions)
        except IndexError:
            pass
    # Reshape the flattened mask image array into the original shape (transposed)
    return np.reshape( mask_img, shape ).T


def encode_rle(mask_img):
    """
    Function to encode a binary mask image using RLE (Run-Length Encoding).

    Parameters:
        mask_img (numpy.ndarray): The binary mask image to be encoded.

    Returns:
        str: The RLE encoded pixel values.
    """
    # Flatten the mask image and convert it to a list of integers
    mask_flat = mask_img.T.flatten()

    # Initialize variables
    rle = []
    count = 0
    current_pixel = -1

    # Iterate through the flattened mask image
    for i, pixel in enumerate( mask_flat ):
        if pixel == 1.:  # Check if the pixel value is 1. (indicating object presence)
            if current_pixel == -1:
                current_pixel = i
                count = 1
            else:
                count += 1
        else:
            if count > 0:
                rle.extend( [current_pixel, count] )
            current_pixel = -1
            count = 0

    # Append the last count and pixel value to the RLE list if count is non-zero
    if count > 0:
        rle.extend( [current_pixel, count] )

    if len( rle ):
        encoded_rle = ' '.join( map( str, rle ) )
    else:
        encoded_rle = pd.NA

    return encoded_rle


def combine_masks(encoded_pixels):
    """
    Function to combine multiple RLE-encoded masks into a single string.

    Parameters:
        encoded_pixels (list): A list of RLE-encoded pixel values.

    Returns:
        str: A string containing the combined RLE-encoded masks.

    """
    masks = ' '.join( map( str, encoded_pixels ) )
    return masks


def process_image(image_id, group):
    """
    Function to process an image group by combining multiple masks into a single combined mask.

    Parameters:
        image_id (str): The ID of the image.
        group (pandas.DataFrame): A DataFrame containing the image group.

    Returns:
        tuple: A tuple containing the image ID, combined mask, and the number of masks in the group.

    """
    # Extract the list of encoded pixels from the DataFrame group
    encoded_pixels = group['EncodedPixels'].tolist()

    # Check if all encoded pixels are not null
    if np.all( pd.notna( encoded_pixels ) ):
        # Combine the masks into a single string
        combined_mask = combine_masks( encoded_pixels )
        # Return the image ID, combined mask, and number of masks
        return image_id, combined_mask, len( group )
    else:
        # If any encoded pixels are null, return None values
        return image_id, None, 0

def calculate_ship_area(encoded_rle, shape):
    """
    Function to calculate the area (percentage of total pixels) of a ship from its RLE encoded mask.

    Parameters:
        encoded_rle (str): The RLE encoded pixel values.
        shape (tuple): The shape (height, width) of the binary mask image.

    Returns:
        float: The area (percentage of total pixels) of the ship.
    """
    total_pixels = shape[0] * shape[1]

    # Decode RLE to get the binary mask image
    mask_img = decode_rle(encoded_rle, shape)

    # Calculate area as a percentage of total pixels
    area_percentage = (np.sum(mask_img) / total_pixels)

    return area_percentage


def dice_coefficient(y_true, y_pred):
    """
    Calculate the Dice coefficient, a metric used for evaluating segmentation performance.

    Args:
        y_true (tensorflow.Tensor): True binary labels.
        y_pred (tensorflow.Tensor): Predicted binary labels.

    Returns:
        float: Dice coefficient value.
    """
    smooth = 1e-15

    y_true_f = tf.keras.backend.flatten( y_true )
    y_pred_f = tf.keras.backend.flatten( y_pred )

    y_true_f = tf.cast( y_true_f, tf.float32 )
    intersection = tf.keras.backend.sum( y_true_f * y_pred_f )
    return (2. * intersection + smooth) / (tf.keras.backend.sum( y_true_f ) + tf.keras.backend.sum( y_pred_f ) + smooth)


def dice_loss(y_true, y_pred):
    """
    Calculate the Dice loss, which is 1 minus the Dice coefficient.

    Args:
        y_true (tensorflow.Tensor): True binary labels.
        y_pred (tensorflow.Tensor): Predicted binary labels.

    Returns:
        float: Dice loss value.
    """
    return 1. - dice_coefficient( y_true, y_pred )


def conv_block(inputs, num_filters):
    """
    Convolutional block consisting of two convolutional layers with batch normalization and ReLU activation.

    Args:
        inputs (tensorflow.Tensor): Input tensor.
        num_filters (int): Number of filters for convolutional layers.

    Returns:
        tensorflow.Tensor: Output tensor.
    """
    x = Conv2D( num_filters, 3, padding="same" )( inputs )
    x = BatchNormalization()( x )
    x = Activation( "relu" )( x )

    x = Conv2D( num_filters, 3, padding="same" )( x )
    x = BatchNormalization()( x )
    x = Activation( "relu" )( x )

    return x


def encoder_block(inputs, num_filters):
    """
    Encoder block comprising a convolutional block followed by max pooling.

    Args:
        inputs (tensorflow.Tensor): Input tensor.
        num_filters (int): Number of filters for the convolutional block.

    Returns:
        Tuple[tensorflow.Tensor, tensorflow.Tensor]: Output tensors from the convolutional block and max pooling.
    """
    x = conv_block( inputs, num_filters )
    p = MaxPool2D( (2, 2) )( x )
    return x, p


def decoder_block(inputs, skip_features, num_filters):
    """
    Decoder block consisting of transposed convolution, concatenation with skip connections, and convolutional block.

    Args:
        inputs (tensorflow.Tensor): Input tensor.
        skip_features (tensorflow.Tensor): Skip connection tensor from encoder block.
        num_filters (int): Number of filters for the convolutional block.

    Returns:
        tensorflow.Tensor: Output tensor.
    """
    x = Conv2DTranspose( num_filters, 2, strides=2, padding="same" )( inputs )
    x = Concatenate()( [x, skip_features] )
    x = conv_block( x, num_filters )
    return x


def build_unet(input_shape):
    """
    Function to build the U-Net model architecture.

    Args:
        input_shape (tuple): Shape of the input tensor (height, width, channels).

    Returns:
        tensorflow.keras.Model: U-Net model.
    """
    inputs = Input( input_shape )

    s1, p1 = encoder_block( inputs, 32 )
    s2, p2 = encoder_block( p1, 64 )
    s3, p3 = encoder_block( p2, 128 )
    s4, p4 = encoder_block( p3, 256 )

    b1 = conv_block( p4, 512 )

    d1 = decoder_block( b1, s4, 256 )
    d2 = decoder_block( d1, s3, 128 )
    d3 = decoder_block( d2, s2, 64 )
    d4 = decoder_block( d3, s1, 32 )

    outputs = Conv2D( 1, 1, padding="same", activation="sigmoid" )( d4 )
    model = Model( inputs, outputs, name="UNET" )
    return model

def create_datasets(df):
    """
    This function creates balanced training and validation datasets from a given DataFrame.

    Args:
    - df (pd.DataFrame): DataFrame containing the image data and associated labels.

    Returns:
    - X_train (np.array): Training set image IDs.
    - X_val (np.array): Validation set image IDs.
    - y_train (np.array): Training set masks.
    - y_val (np.array): Validation set masks.
    """

    print( '---Creating dataset-----------------------------------' )

    # Split the DataFrame into images with ships and images without ships
    has_ships = df[df[num_ships_col] > 0]
    no_ships = df[df[num_ships_col] == 0]

    # Undersample images with ships using weighted sampling based on ship area
    has_ships_undersampled = has_ships.sample( n=number_images_with_ships, weights=ship_area_col,
                                               random_state=random_state_num )

    # Undersample images without ships
    no_ships_undersampled = no_ships.sample( n=number_images_without_ships, random_state=random_state_num )

    # Split the undersampled images with ships into training and validation sets
    X_train_pos, X_val_pos, y_train_pos, y_val_pos = train_test_split(
        has_ships_undersampled[id_col],
        has_ships_undersampled[combined_mask_col],
        test_size=percentage_val_split,
        random_state=random_state_num
    )

    # Split the undersampled images without ships into training and validation sets
    X_train_neg, X_val_neg, y_train_neg, y_val_neg = train_test_split(
        no_ships_undersampled[id_col],
        no_ships_undersampled[combined_mask_col],
        test_size=percentage_val_split,
        random_state=random_state_num
    )

    # Combine positive (with ships) and negative (without ships) samples for training and validation sets
    X_train = np.concatenate( (X_train_pos, X_train_neg) )
    X_val = np.concatenate( (X_val_pos, X_val_neg) )
    y_train = np.concatenate( (y_train_pos, y_train_neg) )
    y_val = np.concatenate( (y_val_pos, y_val_neg) )

    # Shuffle the training and validation sets to ensure random distribution
    X_train, y_train = shuffle( X_train, y_train, random_state=random_state_num )
    X_val, y_val = shuffle( X_val, y_val, random_state=random_state_num )

    return X_train, X_val, y_train, y_val


def train_model(X_train, X_val, y_train, y_val):
    """
    This function builds, compiles, and trains a U-Net model using the provided training and validation datasets.

    Args:
    - X_train (np.array): Training set image IDs.
    - X_val (np.array): Validation set image IDs.
    - y_train (np.array): Training set masks.
    - y_val (np.array): Validation set masks.

    Returns:
    - history (tf.keras.callbacks.History): Training history object.
    """
    # Create U-Net model
    model = build_unet( (height, width, num_channels) )
    print('---Model Training-----------------------------------')

    # Register custom metrics and loss functions
    tf.keras.utils.get_custom_objects()['dice_coefficient'] = dice_coefficient
    tf.keras.utils.get_custom_objects()['dice_loss'] = dice_loss

    # Compile the model with Adam optimizer, dice loss, and dice coefficient metric
    model.compile( optimizer=tf.keras.optimizers.Adam( learning_rate=lr ),
                   loss=[dice_loss],
                   metrics=[dice_coefficient] )

    # Define checkpoint to save the best model based on validation loss
    checkpoint = ModelCheckpoint( filepath_keras, monitor='val_loss', verbose=1, save_best_only=True )

    # Define early stopping to halt training when validation loss stops improving
    early_stopping = EarlyStopping( monitor='val_loss', patience=patience, restore_best_weights=True )

    # Calculate steps per epoch for training and validation
    steps_per_epoch = len( X_train ) // batch_size
    validation_steps = len( X_val ) // batch_size

    # Train the model and save the training history
    history = model.fit(
        data_generator( X_train, y_train, batch_size ),
        steps_per_epoch=steps_per_epoch,
        epochs=num_epochs,
        validation_data=data_generator( X_val, y_val, batch_size ),
        validation_steps=validation_steps,
        callbacks=[checkpoint, early_stopping]
    )

    # Evaluate the final model on the validation set
    loss, dice_score = model.evaluate( data_generator( X_val, y_val, batch_size ), steps=validation_steps )
    print( "Validation loss:", loss )
    print( "Dice score:", dice_score )

    return history

def main():
    # Read the dataset
    df = pd.read_csv(train_dir_csv)
    # Combine the dataset by ImageId
    grouped_data = df.groupby(id_col)
    # Process each group and collect results
    processed_results = [process_image(image_id, group) for image_id, group in grouped_data]
    result_df = pd.DataFrame(processed_results, columns=[id_col, combined_mask_col, num_ships_col] )
    # Calculate the area of ships on each image
    result_df[ship_area_col] = result_df.apply(
        lambda row: calculate_ship_area(row[combined_mask_col], shape=(height, width)), axis=1)
    # Create datasets
    X_train, X_val, y_train, y_val = create_datasets(result_df)
    # Train the model and get the history of training
    history = train_model(X_train, X_val, y_train, y_val)


if __name__ == "__main__":
    main()