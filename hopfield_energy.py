import os

import torch

from utils.energy import BorderEnergy


def hopfield_loss(emb_in, emb_out, query=None, beta_a=4, beta_b=4):
    border_energy = BorderEnergy(a=emb_in, b=emb_out, beta_a=beta_a, beta_b=beta_b)
    if query is None:
        query = torch.concat([emb_in, emb_out], dim=0)
    energies, info_dict = border_energy(query, return_dict=True)
    return -torch.mean(energies), {'border_energies': energies, **info_dict}


def energy_loss(logits_in, logits_out, m_in=-23, m_out=-5):  # implementation of https://arxiv.org/pdf/2010.03759.pdf
    in_energy = -torch.logsumexp(logits_in, dim=-1)
    out_energy = -torch.logsumexp(logits_out, dim=-1)
    in_losses = torch.maximum(torch.tensor(0.), in_energy - m_in) ** 2
    out_losses = torch.maximum(torch.tensor(0.), m_out - out_energy) ** 2
    loss = torch.mean(in_losses) + torch.mean(out_losses)
    return loss, {'in_energy': in_energy, 'out_energy': out_energy, 'in_losses': in_losses, 'out_losses': out_losses}


def msp_loss(logits_out):  # implementation of https://arxiv.org/pdf/1812.04606.pdf
    return -(logits_out.mean(-1) - torch.logsumexp(logits_out, dim=-1)).mean(), {}

if __name__ == '__main__':
    root_folder = '/data01/gio/ctrl/data/embeddings_best'

    memory = {}
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.pt'):
                filepath = os.path.join(dirpath, filename)
                class_name = filename.split('_')[1].split('.')[0]
                tensor = torch.load(filepath)
                if class_name in memory:
                    memory[class_name].append(tensor)
                else:
                    memory[class_name] = [tensor]
    for class_name, class_tensors in memory.items():
        memory[class_name] = torch.stack(class_tensors)

    mean_energies, energies_dict = hopfield_loss(memory["noclass"].squeeze(), memory["noclass"].squeeze(), memory["cigarette"].squeeze())

    print(mean_energies, energies_dict)