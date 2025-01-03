import mlflow.pyfunc

# Load the model
model = mlflow.pyfunc.load_model("models:/plugin_model/1")

# Print the schema
print(model.metadata.get_input_schema())
