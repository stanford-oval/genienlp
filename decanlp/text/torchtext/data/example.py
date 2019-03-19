import csv
import sys
import json

import six


def intern_strings(x):
    if isinstance(x, (list, tuple)):
        r = []
        for y in x:
            if isinstance(y, str):
                r.append(sys.intern(y))
            else:
                r.append(y)
        return r
    return x


class Example(object):
    """Defines a single training or test example.

    Stores each column of the example as an attribute.
    """

    @classmethod
    def fromJSON(cls, data, fields, **kwargs):
        return cls.fromdict(json.loads(data), fields, **kwargs)

    @classmethod
    def fromdict(cls, data, fields, **kwargs):
        ex = cls()
        for key, vals in fields.items():
            if key not in data:
                raise ValueError("Specified key {} was not found in "
                                 "the input data".format(key))
            if vals is not None:
                if not isinstance(vals, list):
                    vals = [vals]
                for val in vals:
                    name, field = val
                    setattr(ex, name, intern_strings(field.preprocess(data[key], field_name=name, **kwargs)))
        return ex

    @classmethod
    def fromTSV(cls, data, fields, **kwargs):
        return cls.fromlist(data.split('\t'), fields, **kwargs)

    @classmethod
    def fromCSV(cls, data, fields, **kwargs):
        data = data.rstrip("\n")
        # If Python 2, encode to utf-8 since CSV doesn't take unicode input
        if six.PY2:
            data = data.encode('utf-8')
        # Use Python CSV module to parse the CSV line
        parsed_csv_lines = csv.reader([data])

        # If Python 2, decode back to unicode (the original input format).
        if six.PY2:
            for line in parsed_csv_lines:
                parsed_csv_line = [six.text_type(col, 'utf-8') for col in line]
                break
        else:
            parsed_csv_line = list(parsed_csv_lines)[0]
        return cls.fromlist(parsed_csv_line, fields, **kwargs)

    @classmethod
    def fromlist(cls, data, fields, **kwargs):
        ex = cls()
        for (name, field), val in zip(fields, data):
            if field is not None:
                if isinstance(val, six.string_types):
                    val = val.rstrip('\n')
                setattr(ex, name, intern_strings(field.preprocess(val, **kwargs)))
        return ex

    @classmethod
    def fromtree(cls, data, fields, subtrees=False, **kwargs):
        try:
            from nltk.tree import Tree
        except ImportError:
            print("Please install NLTK. "
                  "See the docs at http://nltk.org for more information.")
            raise
        tree = Tree.fromstring(data)
        if subtrees:
            return [cls.fromlist(
                [' '.join(t.leaves()), t.label()], fields) for t in tree.subtrees()]
        return cls.fromlist([' '.join(tree.leaves()), tree.label()], fields, **kwargs)
