import dill


def save_model_artifacts(path, ml_artifact):
    with open(path, "wb") as dill_outfile:
        dill.dump(ml_artifact, dill_outfile)
        print('Saved ml artifact at location:', path)


def load_model_artifacts(path):
    with open(path, "rb") as dill_infile:
        ml_artifact = dill.load(dill_infile)
        print('Loaded ml artifact from location:', path)
    return ml_artifact