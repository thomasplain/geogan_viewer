import torch
import torch.nn as nn
import itertools

from geo_gan.models import networks

import re

class Model(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.model = None

        torch.cuda.init()

    def opts_from_slurm(self, filename):
        text = open(filename).read().splitlines()
        start_line = [i for i, line in enumerate(text[:10]) if '- Options -' in line][0]
        end_line = next((i for i, line in enumerate(text) if '- End -' in line))
        opts_text = text[start_line + 1:end_line]
        dataroot_line_no, dataroot_line = next(((i, line) for i, line in enumerate(opts_text) if 'dataroot: ' in line))
        opts_text.pop(dataroot_line_no)
        opts_text.insert(0, dataroot_line)

        max_dataset_index = opts_text.index('max_dataset_size: inf')
        if max_dataset_index:
            opts_text.pop(max_dataset_index)

        return dict(line.split(': ') for line in opts_text)


    def arch_from_slurm(self, filename):
        self.opts_dict = self.opts_from_slurm(filename)
        input_nc = int(self.opts_dict['input_nc'])
        output_nc = int(self.opts_dict['output_nc'])
        ngf = int(self.opts_dict['ngf'])
        which_model_netG = self.opts_dict['which_model_netG']
        norm = self.opts_dict['norm']
        use_dropout = not (self.opts_dict['no_dropout'] == 'True')
        init = self.opts_dict['init_type']

        with_BCE = self.opts_dict['with_BCE'] == 'True'

        if with_BCE:
            output_nc += 1

        cuda_device = torch.device('cuda')


        self.model = networks.define_G(input_nc, output_nc, ngf, which_model_netG,
                                       norm, use_dropout, init, gpu_ids=[0])



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
                #     next_line = filetext[i+1]
                #     leading_ws = len(next_line) - len(next_line.lstrip())
                #     # Search for next block with same indentation - this is the matching bracket
                #     # that we want to remove as well
                #     j = i+2
                #     for line2 in filetext[i+2:]:
                #         if len(line2) - len(line2.lstrip()) == leading_ws:
                #             break
                #
                #         j += 1
                #
                #     filetext.pop(j)
                #     filetext.pop(i)
                #
                #     continue
                    block_name = 'Sequential'
                    line = block_name + '(' + '('.join(line.split(': ')[-1].split('(')[1:])

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
            print(arch_text)
            self.model = eval(arch_text)

if __name__ == '__main__':
    m = Model()
    m.arch_from_file('arch.txt')
    print(m.model)
