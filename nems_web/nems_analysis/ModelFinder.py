"""Defines the ModelFinder class, used for parsing modeltree strings."""
import logging
import ast
import re

log = logging.getLogger(__name__)


class ModelFinder():
    """Converts a modeltree string from Analysis into a list of modelnames.

    Invokes a series of internal methods on any modeltree string passed to
    its constructor.
    First, string_to_nested_list cleans up the string to match the syntax for
        nested python lists, and interprets the string as a list using
        ast.literal_eval.
    Then, traverse_nested recursively finds all combinations of the strings
        (i.e. 'keywords') inside the lists and stores them in self.comboArray.
    Finally, array_to_list converts each list of strings contained in
        self.comboArray into a string concatenation of the list separated
        with underscores, and stores a list of the strings in self.modellist.
    self.modellist is ultimately used by the update_models view function in
        nems_analysis.views.

    Arguments:
    ----------
    modelstring : string
        A model tree specified in a Analysis entry, resembling a nested
        list, ex: {'an','example',{'tree','string'}}

    See Also:
    ---------
    _Analysis : rebuild_model_tree, keyword_combos

    """

    def __init__(self, modelstring='', sep='-'):
        self.modelstring = modelstring
        self.sep = sep
        # As soon as modelstring is passed, go ahead and
        # parse the string into an array then to a list so that
        # self.modellist attribute can be retrieved by nemsweb.py
        self.nestedlist = self.string_to_nested_list()
        self.comboArray = []
        self.traverse_nested(self.nestedlist, [])
        self.modellist = self.array_to_list(self.comboArray)

    def string_to_nested_list(self):
        """Clean up the modeltree string.

        Replace curly braces with brackets, remove whitespace, add/remove
        commas or excess brackets, fix other syntax issues as they come up.
        Then use ast.literal_eval to interpret the (hopefully) syntactically
        valid string as a nested list.

        """

        # TODO: May not be a complete list of fixes - have been playing
        # whack-a-mole with syntax errors so far. ast.literal_eval is
        # very picky about commas, brackets etc, but still seems more
        # efficient than writing a new function for evaluating the string.
        # If a modelstring is found that won't work with this,
        # chances are another s.replace(...) is needed to fix
        # some syntax error.
        s = self.modelstring.replace('{', '[')
        s = s.replace('}', ']')
        s = s.replace(" ", "")
        s = s.replace('],]', ']]')
        s = s.replace('  ', ' ')
        s = s.replace("]'", "],'")
        s = s.replace("][", "],[")
        s = s.replace("'[", "',[")

        # Insert comma between adjacent quotation marks unless it's an empty
        # string
        r = re.compile(r"(?P<ONE>\w)''(?P<TWO>\w)")
        s = r.sub("\g<ONE>','\g<TWO>", s)
        # Insert comma between empty string and regular string (empty first)
        r = re.compile(r"(?P<ONE>'')(?P<TWO>'\w)")
        s = r.sub("\g<ONE>,\g<TWO>", s)
        # Insert comma between empty string and regular string (empty second)
        r = re.compile(r"(?P<ONE>'\w)(?P<TWO>'')")
        s = r.sub("\g<ONE>,\g<TWO>", s)

        try:
            nestedlist = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            print(
                """
                \n\n ast.literal_eval has issues with the string format for
                this modeltree:\n\n
                %s\n\n
                Or modelstring was not present in Analysis
                """ % self.modelstring
                )
            return ([
                'ast', 'eval', 'did', 'not', 'work', 'for', 'this',
                'analysis', 'or', 'model', 'string', 'not', 'present',
                ])

        return nestedlist

    def traverse_nested(self, nestedlist, path):
        """Recursively form an array of all combinations of strings
        (or lists of strings) contained in a nested list.

        """

        # If nestedList is empty, recursion reached a 'leaf.'
        if len(nestedlist) == 0:
            self.comboArray.append(path)
        elif type(nestedlist) is list:
            head = nestedlist[0]
            tail = nestedlist[1:]
            # If first element of nestedList is a string, append and move on.
            if type(head) is str:
                self.traverse_nested(tail, (path+[head]))
            # If it's a list, iteratively append each item to a new path
            # and traverse the rest of the list on each path
            elif type(head) is list:
                for l in range(len(head)):
                    if type(head[l]) is list:
                        self.traverse_nested(tail, path+head[l])
                    elif type(head[l]) is str:
                        self.traverse_nested(tail, path+[head[l]])
        # If nestedList isn't a list, something didn't work right.
        else:
            # TODO: Need some kind of error here?
            pass

        return

    def array_to_list(self, array):
        """Parse each list of strings in self.comboArray into a string
        concatenation of the list separated by underscores.

        """
        # Ignore blank strings when joining underscores
        models = [self.sep.join(filter(None, c)) for c in self.comboArray]

        return models
