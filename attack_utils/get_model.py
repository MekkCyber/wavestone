from dl_models import labeler_cnn

model = labeler_cnn.create_model()


checkpoint_path = "checkpoints/labeler/training_1/cp.ckpt"

model.load_weights(checkpoint_path).expect_partial()

ds_train, _, ds_test = labeler_cnn.get_dataset()

loss, acc = model.evaluate(ds_test, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))