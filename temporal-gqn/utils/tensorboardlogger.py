
import json
import re

class TensorBoardLogger:
    writer = None

    def log_class_attributes(self, inst, tag="attributes"):

        if tensorboardlogger.writer:
            opt_dict = {"class" : inst.__class__.__name__}
            for arg in vars(inst):
                if not arg.startswith("_"):
                    opt_dict[str(arg)] = str(getattr(inst, arg))

            opt_dict_json = str(json.dumps(opt_dict, indent=4)).replace("\n","  \n")
            self.writer.add_text(tag, opt_dict_json)

tensorboardlogger = TensorBoardLogger()

def lists_to_tikz(X, Y):
    assert len(X) == len(Y), "Sizes of arguments x and y should be equal."
    points = [(x,y) for x,y in zip(X,Y)]
    return re.sub("\), \(", ")(", str(points))[1:-1]