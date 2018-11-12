import torch
import torch.nn as nn

import re

class Model(nn.Module):


    def arch_from_file(self, filename):
        with open(filename) as arch_file:
            filetext = arch_file.read().splitlines()
            filetext = [line for line in filetext if line != '']    # Remove any blank lines

            i = 0
            while i < len(filetext):
                line = filetext[i]

                # If the network block is not a pytorch default, we just remove that line -
                # The following lines define the block in terms of Pytorch blocks anyway
                block_name = line.split(': ')[-1].split('(')[0]
                if block_name not in dir(nn) and line.strip() != ')':
                    next_line = filetext[i+1]
                    leading_ws = len(next_line) - len(next_line.lstrip())
                    # Search for next block with same indentation - this is the matching bracket
                    # that we want to remove as well
                    j = i+2
                    for line2 in filetext[i+2:]:
                        if len(line2) - len(line2.lstrip()) == leading_ws:
                            break

                        j += 1

                    filetext.pop(j)
                    filetext.pop(i)

                    continue

                # Remove weird line header, and add an 'nn.' so the module can be imported
                line = line.split(': ')[-1].strip()

                if block_name in dir(nn):
                    line = 'nn.' + line

                filetext[i] = line

                i += 1

            # Make one long string so we can use regex to substitute
            arch_text = ''.join(filetext)

            def insert_comma(matchobj):
                return '),' + matchobj.group(1)

            # Just joining as above doesn't handle sequential modules correctly
            # But we don't want to put commas between brackets
            arch_text = re.sub('\)([^\),$][\w]*)', insert_comma, arch_text)

            # Replace just the keyword with a keyword-arg pair so we don't get an error
            arch_text = re.sub('inplace', 'inplace=True', arch_text)

            self.model = eval(arch_text)


