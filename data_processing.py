import os
import urllib
import urllib2
import hashlib
from scipy import misc
from PIL import Image
import numpy as np


def name_file(processed, male, actor, line):
    #names the downloaded image file, specifying its location to be saved
    first = processed + '/'

    if male ==0:
        second = 'Actresses/'
    else:
        second = 'Actors/'

    third = actor + '/'
    fourth = line + '.jpg'

    return first+second+third+fourth

def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray / 255.

def process_image(raw_name, processed_name, x1, y1, x2, y2, size):
    # Reads raw image and converts it to grayscale
    img = misc.imread(raw_name)
    #gray_img = rgb2gray(img)

    # Crops the image and saves it to its new location
    misc.imsave(processed_name, mg[y1:y2, x1:x2]) #changed gray_img -> img

    # Opens cropped image, resizes it, saves it in same location
    big_img = Image.open(processed_name)
    sml_img = big_img.resize((size, size), Image.ANTIALIAS)
    sml_img.save(processed_name)

    return

def download_and_process_images(MorF, size):

    actors_or_actresses= ['actresses.txt', 'actors.txt']
    text_file = open(actors_or_actresses[MorF], 'r')
    pic_counter = 0

    actors = ['Alec Baldwin', 'Bill Hader', 'Steve Carell', 'Angie Harmon','Peri Gilpin', 'Lorraine Bracco'] #took out lorraine bracco

    for line in text_file:
        text = line.split()
        actor = text[0] + ' ' + text[1]
        if actor not in actors:
            continue
        try:
            if actor != prev_actor:
                pic_counter = 0
        except:
            prev_actor = actor

        prev_actor = actor

        # Names raw image from url
        raw_name = name_file('Raw', MorF, actor, str(pic_counter))
        #Checks the hash of the image and compares it against the hash in the text file to
        #see if the image has been changed
        try:
            request = urllib2.urlopen(text[4])
            hashcheck = request.read()
        except:
            continue
        if hashlib.sha256(hashcheck).hexdigest() != text[6]:
            continue

        # Names new processed file and reads bounding box coordinates from text file
        processed_name = name_file('Processed227', MorF, actor, str(pic_counter))
        crop = text[5].split(",")
        x1, y1, x2, y2 = int(crop[0]), int(crop[1]), int(crop[2]), int(crop[3])

        #Downloads the image from the URL then
        # Converts image to grayscale, crops face, resizes image to 32 x 32
        try:
            urllib.urlopen(text[4]) #didn't need to redownload, this was changed to open to check if files existed
            process_image(raw_name, processed_name, x1, y1, x2, y2, size)
            pic_counter += 1
        except:
            continue


def partition_images(MorF):
        np.random.seed(58)
        #actors = ['Alec Baldwin', 'Bill Hader', 'Daniel Radcliffe', 'Gerard Butler', 'Michael Vartan', 'Steve Carell']
        #actresses = ['America Ferrera', 'Angie Harmon', 'Fran Drescher', 'Kristin Chenoweth', 'Lorraine Bracco', 'Peri Gilpin']
        actors = ['Alec Baldwin', 'Bill Hader', 'Steve Carell']
        actresses = ['Angie Harmon', 'Lorraine Bracco','Peri Gilpin']
        if MorF == 0:
            acts = actresses
            act = 'Actresses/'
        else:
            acts = actors
            act = 'Actors/'
        sets = ['Training227', 'Validation227', 'Test227']

        for actor in acts:

            dir = os.getcwd() + '/Processed/' + act + actor
            files_in_directory = os.listdir(dir)
            num_of_pics = len(files_in_directory)
            rand_array = np.random.permutation(num_of_pics)


            rand_array_counter = 0

            for set in sets:
                if set == 'Training227':
                    if actor == 'Peri Gilpin':
                        x = 55
                    else:
                        x = 70
                if set == 'Validation227':
                    x = 10
                if set == 'Test227':
                    x = 20

                for i in range(x):
                    partition_name = name_file(set, MorF, actor, str(i))
                    processed_name = name_file('Processed227', MorF, actor, str(rand_array[rand_array_counter]))
                    img = Image.open(processed_name)
                    img.save(partition_name)
                    rand_array_counter += 1

'''def process_all_images(MorF):
    actors = ['Alec Baldwin', 'Bill Hader', 'Steve Carell']
    actresses = ['Angie Harmon', 'Lorraine Bracco', 'Peri Gilpin']
    if MorF == 0:
        acts = actresses
        act = 'Actresses/'
    else:
        acts = actors
        act = 'Actors/'
    sets = ['Training227', 'Validation227', 'Test227']
    for actor in acts:
        dir_r = os.getcwd() + '/Raw/' + act + actor
        dir_p = os.getcwd() + '/Processed227/' + act + actor

        for set in sets:
            if set == 'Training227':
                if actor == 'Peri Gilpin':
                    x = 55
                else:
                    x = 70
            if set == 'Validation227':
                x = 10
            if set == 'Test227':
                x = 20'''

if __name__ == '__main__':
    #To change the files downloaded, change MorF. 1 corresponds to actors, 0 corresponds to actresses

    #this has been edited in order to output 227x227 rgb pictures instead of 32x32 black and white, downloading
    #has also been turned of since files have already been acquired
    for i in range(2):
        MorF = 1
        size = 227  #change the number of pixels the images are resized to
        download_and_process_images(MorF, size)
        partition_images(MorF)
