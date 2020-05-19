import face_recognition

known_image = face_recognition.load_image_file("sister.jpg")

# Get the face encodings for the known images
known_face_encoding = face_recognition.face_encodings(known_image)[0]
# biden_face_encoding = face_recognition.face_encodings(known_biden_image)[0]
# print(known_face_encoding)

known_encodings = [known_face_encoding]

# Load a test image and get encondings for it
image_to_test = face_recognition.load_image_file("2.jpg")
image_to_test_encoding = face_recognition.face_encodings(image_to_test)[0]

# See how far apart the test image is from the known faces
face_distances = face_recognition.face_distance(known_encodings, image_to_test_encoding)

for i, face_distance in enumerate(face_distances):
    print("The test image has a distance of {:.2} from known image ".format(face_distance, i))
    print('Similarity percentage: {} %'.format((1-face_distance)*100,i))
    print("- With a normal cutoff of 0.75, would the test image match the known image? {}".format(face_distance < 0.75))
    print("- With a very strict cutoff of 0.5, would the test image match the known image? {}".format(face_distance < 0.5))
    print()



