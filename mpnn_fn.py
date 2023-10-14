import argparse
import os.path

def main(args):

    import json, time, os, sys, glob
    import shutil
    import warnings
    import numpy as np
    import torch
    from torch import optim
    from torch.utils.data import DataLoader
    from torch.utils.data.dataset import random_split, Subset
    import copy
    import torch.nn as nn
    import torch.nn.functional as F
    import random
    import os.path
    import subprocess
    
    from protein_mpnn_utils import loss_nll, loss_smoothed, gather_edges, gather_nodes, gather_nodes_t, cat_neighbors_nodes, _scores, _S_to_seq, tied_featurize, parse_PDB, parse_fasta
    from protein_mpnn_utils import StructureDataset, StructureDatasetPDB, ProteinMPNN

    if args.seed:
        seed=args.seed
    else:
        seed=int(np.random.randint(0, high=999, size=1, dtype=int)[0])

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)   
    
    hidden_dim = 128
    num_layers = 3 
    
    import pickle
    def save_dict_to_pickle(dictionary, filename):
        with open(filename, 'wb') as file:
            pickle.dump(dictionary, file)

    if True:
        model_folder_path = "/root/ProteinMPNN/vanilla_model_weights/"
        model_folder_path_s = "/root/ProteinMPNN/soluble_model_weights/"
        model_folder_path_c = "/root/ProteinMPNN/ca_model_weights/"
        if model_folder_path[-1] != '/':
            model_folder_path = model_folder_path + '/'
    else: 
        file_path = os.path.realpath(__file__)
        k = file_path.rfind("/")
        if args.ca_only:
            print("Using CA-ProteinMPNN!")
            model_folder_path = file_path[:k] + '/ca_model_weights/'
            if args.use_soluble_model:
                print("WARNING: CA-SolubleMPNN is not available yet")
                sys.exit()
        else:
            if args.use_soluble_model:
                print("Using ProteinMPNN trained on soluble proteins only!")
                model_folder_path = file_path[:k] + '/soluble_model_weights/'
            else:
                model_folder_path = file_path[:k] + '/vanilla_model_weights/'

    # checkpoint_path = model_folder_path + f'{args.model_name}.pt'
    checkpoint_path_1 = model_folder_path + 'v_48_002.pt'
    checkpoint_path_2 = model_folder_path + 'v_48_010.pt'
    checkpoint_path_3 = model_folder_path + 'v_48_020.pt'
    checkpoint_path_4 = model_folder_path + 'v_48_030.pt'
    checkpoint_path_5 = model_folder_path_s + 'v_48_010.pt'
    checkpoint_path_6 = model_folder_path_s + 'v_48_020.pt'
    checkpoint_path_7 = model_folder_path_c + 'v_48_002.pt'
    checkpoint_path_8 = model_folder_path_c + 'v_48_010.pt'
    checkpoint_path_9 = model_folder_path_c + 'v_48_020.pt'

    folder_for_outputs = args.out_folder
    
    NUM_BATCHES = args.num_seq_per_target//args.batch_size
    BATCH_COPIES = args.batch_size
    temperatures = [float(item) for item in args.sampling_temp.split()]
    omit_AAs_list = args.omit_AAs
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    alphabet_dict = dict(zip(alphabet, range(21)))    
    print_all = args.suppress_print == 0 
    omit_AAs_np = np.array([AA in omit_AAs_list for AA in alphabet]).astype(np.float32)
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    if os.path.isfile(args.chain_id_jsonl):
        with open(args.chain_id_jsonl, 'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            chain_id_dict = json.loads(json_str)
    else:
        chain_id_dict = None
        if print_all:
            print(40*'-')
            print('chain_id_jsonl is NOT loaded')
        
    if os.path.isfile(args.fixed_positions_jsonl):
        with open(args.fixed_positions_jsonl, 'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            fixed_positions_dict = json.loads(json_str)
    else:
        if print_all:
            print(40*'-')
            print('fixed_positions_jsonl is NOT loaded')
        fixed_positions_dict = None
    
    
    if os.path.isfile(args.pssm_jsonl):
        with open(args.pssm_jsonl, 'r') as json_file:
            json_list = list(json_file)
        pssm_dict = {}
        for json_str in json_list:
            pssm_dict.update(json.loads(json_str))
    else:
        if print_all:
            print(40*'-')
            print('pssm_jsonl is NOT loaded')
        pssm_dict = None
    
    
    if os.path.isfile(args.omit_AA_jsonl):
        with open(args.omit_AA_jsonl, 'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            omit_AA_dict = json.loads(json_str)
    else:
        if print_all:
            print(40*'-')
            print('omit_AA_jsonl is NOT loaded')
        omit_AA_dict = None
    
    
    if os.path.isfile(args.bias_AA_jsonl):
        with open(args.bias_AA_jsonl, 'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            bias_AA_dict = json.loads(json_str)
    else:
        if print_all:
            print(40*'-')
            print('bias_AA_jsonl is NOT loaded')
        bias_AA_dict = None
    
    
    if os.path.isfile(args.tied_positions_jsonl):
        with open(args.tied_positions_jsonl, 'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            tied_positions_dict = json.loads(json_str)
    else:
        if print_all:
            print(40*'-')
            print('tied_positions_jsonl is NOT loaded')
        tied_positions_dict = None

    
    if os.path.isfile(args.bias_by_res_jsonl):
        with open(args.bias_by_res_jsonl, 'r') as json_file:
            json_list = list(json_file)
    
        for json_str in json_list:
            bias_by_res_dict = json.loads(json_str)
        if print_all:
            print('bias by residue dictionary is loaded')
    else:
        if print_all:
            print(40*'-')
            print('bias by residue dictionary is not loaded, or not provided')
        bias_by_res_dict = None
   

    if print_all: 
        print(40*'-')
    bias_AAs_np = np.zeros(len(alphabet))
    if bias_AA_dict:
            for n, AA in enumerate(alphabet):
                    if AA in list(bias_AA_dict.keys()):
                            bias_AAs_np[n] = bias_AA_dict[AA]
    
    if False:
        pdb_dict_list = parse_PDB(args.pdb_path, ca_only=args.ca_only)
        dataset_valid = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=args.max_length)
        all_chain_list = [item[-1:] for item in list(pdb_dict_list[0]) if item[:9]=='seq_chain'] #['A','B', 'C',...]
        if args.pdb_path_chains:
            designed_chain_list = [str(item) for item in args.pdb_path_chains.split()]
        else:
            designed_chain_list = all_chain_list
        fixed_chain_list = [letter for letter in all_chain_list if letter not in designed_chain_list]
        chain_id_dict = {}
        chain_id_dict[pdb_dict_list[0]['name']]= (designed_chain_list, fixed_chain_list)
    else:
        dataset_valid = StructureDataset(args.jsonl_path, truncate=None, max_length=args.max_length, verbose=print_all)

    # checkpoint = torch.load(checkpoint_path, map_location=device) 
    checkpoint_1 = torch.load(checkpoint_path_1, map_location=device) 
    checkpoint_2 = torch.load(checkpoint_path_2, map_location=device) 
    checkpoint_3 = torch.load(checkpoint_path_3, map_location=device) 
    checkpoint_4 = torch.load(checkpoint_path_4, map_location=device) 
    checkpoint_5 = torch.load(checkpoint_path_5, map_location=device) 
    checkpoint_6 = torch.load(checkpoint_path_6, map_location=device) 
    checkpoint_7 = torch.load(checkpoint_path_7, map_location=device) 
    checkpoint_8 = torch.load(checkpoint_path_8, map_location=device) 
    checkpoint_9 = torch.load(checkpoint_path_9, map_location=device) 
    noise_level_print = checkpoint_1['noise_level']

    def load_model(checkpoint):
        model = ProteinMPNN(ca_only=args.ca_only, 
                              num_letters=21, 
                              node_features=hidden_dim, 
                              edge_features=hidden_dim, 
                              hidden_dim=hidden_dim, 
                              num_encoder_layers=num_layers, 
                              num_decoder_layers=num_layers, 
                              augment_eps=args.backbone_noise, 
                              k_neighbors=checkpoint['num_edges'])
        model.to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        return model
    
    def load_model_Ca(checkpoint):
        model = ProteinMPNN(ca_only=True, 
                              num_letters=21, 
                              node_features=hidden_dim, 
                              edge_features=hidden_dim, 
                              hidden_dim=hidden_dim, 
                              num_encoder_layers=num_layers, 
                              num_decoder_layers=num_layers, 
                              augment_eps=args.backbone_noise, 
                              k_neighbors=checkpoint['num_edges'])
        model.to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        return model

    model_1 = load_model(checkpoint_1)
    model_2 = load_model(checkpoint_2)
    model_3 = load_model(checkpoint_3)
    model_4 = load_model(checkpoint_4)
    model_5 = load_model(checkpoint_5)
    model_6 = load_model(checkpoint_6)
    model_7 = load_model_Ca(checkpoint_7)
    model_8 = load_model_Ca(checkpoint_8)
    model_9 = load_model_Ca(checkpoint_9)

 
    # Build paths for experiment
    base_folder = folder_for_outputs
    if base_folder[-1] != '/':
        base_folder = base_folder + '/'
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    
    if not os.path.exists(base_folder + 'seqs'):
        os.makedirs(base_folder + 'seqs')
    
    if args.save_score:
        if not os.path.exists(base_folder + 'scores'):
            os.makedirs(base_folder + 'scores')

    if args.score_only:
        if not os.path.exists(base_folder + 'score_only'):
            os.makedirs(base_folder + 'score_only')
   

    if args.conditional_probs_only:
        if not os.path.exists(base_folder + 'conditional_probs_only'):
            os.makedirs(base_folder + 'conditional_probs_only')

    if args.unconditional_probs_only:
        if not os.path.exists(base_folder + 'unconditional_probs_only'):
            os.makedirs(base_folder + 'unconditional_probs_only')
 
    if args.save_probs:
        if not os.path.exists(base_folder + 'probs'):
            os.makedirs(base_folder + 'probs') 
    

    count = 0
    # Validation epoch
    with torch.no_grad():
        for ix, protein in enumerate(dataset_valid):

            count += 1 
            batch_clones = [copy.deepcopy(protein) for i in range(BATCH_COPIES)]
            ori_seq ,X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(batch_clones, device, chain_id_dict, fixed_positions_dict, omit_AA_dict, tied_positions_dict, pssm_dict, bias_by_res_dict, ca_only=args.ca_only)
            ori_seq_caa, X_caa, S_caa, mask_caa, _, chain_M_caa, chain_encoding_all_caa, _, _, _, _, chain_M_pos_caa, omit_AA_mask_caa, residue_idx_caa, _, _, pssm_coef_caa, pssm_bias_caa, pssm_log_odds_all_caa, bias_by_res_all_caa, _ = tied_featurize(batch_clones, device, chain_id_dict, fixed_positions_dict, omit_AA_dict, tied_positions_dict, pssm_dict, bias_by_res_dict, ca_only=True)

            pssm_log_odds_mask = (pssm_log_odds_all > args.pssm_threshold).float() #1.0 for true, 0.0 for false
            pssm_log_odds_mask_caa = (pssm_log_odds_all_caa > args.pssm_threshold).float() #1.0 for true, 0.0 for false

            
            randn_1 = torch.randn(chain_M.shape, device=X.device)
            randn_1_caa = torch.randn(chain_M_caa.shape, device=X.device)

            _, mpnn_emb11 = model_1(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1)
            _, mpnn_emb12 = model_2(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1)
            _, mpnn_emb13 = model_3(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1)
            _, mpnn_emb14 = model_4(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1)
            _, mpnn_emb15 = model_5(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1)
            _, mpnn_emb16 = model_6(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1)
            _, mpnn_emb17 = model_7(X_caa, S_caa, mask_caa, chain_M_caa*chain_M_pos_caa, residue_idx_caa, chain_encoding_all_caa, randn_1_caa)
            _, mpnn_emb18 = model_8(X_caa, S_caa, mask_caa, chain_M_caa*chain_M_pos_caa, residue_idx_caa, chain_encoding_all_caa, randn_1_caa)
            _, mpnn_emb19 = model_9(X_caa, S_caa, mask_caa, chain_M_caa*chain_M_pos_caa, residue_idx_caa, chain_encoding_all_caa, randn_1_caa)
                        
            mpnn_emb1 = torch.cat((
                mpnn_emb11,
                mpnn_emb12,
                mpnn_emb13,
                mpnn_emb14,
                mpnn_emb15,
                mpnn_emb16,
                mpnn_emb17,
                mpnn_emb18,
                mpnn_emb19,
            ),dim=-1)   

            for temp in temperatures:
                for j in range(NUM_BATCHES):
                    randn_2 = torch.randn(chain_M.shape, device=X.device)
                    randn_2_caa = torch.randn(chain_M_caa.shape, device=X.device)
                    
                    sample_dict1 = model_1.sample(X, randn_2, S, chain_M, chain_encoding_all, residue_idx, mask=mask, temperature=temp, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=args.pssm_multi, pssm_log_odds_flag=bool(args.pssm_log_odds_flag), pssm_log_odds_mask=pssm_log_odds_mask, pssm_bias_flag=bool(args.pssm_bias_flag), bias_by_res=bias_by_res_all)
                    sample_dict2 = model_2.sample(X, randn_2, S, chain_M, chain_encoding_all, residue_idx, mask=mask, temperature=temp, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=args.pssm_multi, pssm_log_odds_flag=bool(args.pssm_log_odds_flag), pssm_log_odds_mask=pssm_log_odds_mask, pssm_bias_flag=bool(args.pssm_bias_flag), bias_by_res=bias_by_res_all)
                    sample_dict3 = model_3.sample(X, randn_2, S, chain_M, chain_encoding_all, residue_idx, mask=mask, temperature=temp, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=args.pssm_multi, pssm_log_odds_flag=bool(args.pssm_log_odds_flag), pssm_log_odds_mask=pssm_log_odds_mask, pssm_bias_flag=bool(args.pssm_bias_flag), bias_by_res=bias_by_res_all)
                    sample_dict4 = model_4.sample(X, randn_2, S, chain_M, chain_encoding_all, residue_idx, mask=mask, temperature=temp, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=args.pssm_multi, pssm_log_odds_flag=bool(args.pssm_log_odds_flag), pssm_log_odds_mask=pssm_log_odds_mask, pssm_bias_flag=bool(args.pssm_bias_flag), bias_by_res=bias_by_res_all)
                    sample_dict5 = model_5.sample(X, randn_2, S, chain_M, chain_encoding_all, residue_idx, mask=mask, temperature=temp, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=args.pssm_multi, pssm_log_odds_flag=bool(args.pssm_log_odds_flag), pssm_log_odds_mask=pssm_log_odds_mask, pssm_bias_flag=bool(args.pssm_bias_flag), bias_by_res=bias_by_res_all)
                    sample_dict6 = model_6.sample(X, randn_2, S, chain_M, chain_encoding_all, residue_idx, mask=mask, temperature=temp, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=args.pssm_multi, pssm_log_odds_flag=bool(args.pssm_log_odds_flag), pssm_log_odds_mask=pssm_log_odds_mask, pssm_bias_flag=bool(args.pssm_bias_flag), bias_by_res=bias_by_res_all)
                    sample_dict7 = model_7.sample(X_caa, randn_2_caa, S_caa, chain_M_caa, chain_encoding_all_caa, residue_idx_caa, mask=mask_caa, temperature=temp, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos_caa, omit_AA_mask=omit_AA_mask_caa, pssm_coef=pssm_coef_caa, pssm_bias=pssm_bias_caa, pssm_multi=args.pssm_multi, pssm_log_odds_flag=bool(args.pssm_log_odds_flag), pssm_log_odds_mask=pssm_log_odds_mask_caa, pssm_bias_flag=bool(args.pssm_bias_flag), bias_by_res=bias_by_res_all_caa)
                    sample_dict8 = model_8.sample(X_caa, randn_2_caa, S_caa, chain_M_caa, chain_encoding_all_caa, residue_idx_caa, mask=mask_caa, temperature=temp, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos_caa, omit_AA_mask=omit_AA_mask_caa, pssm_coef=pssm_coef_caa, pssm_bias=pssm_bias_caa, pssm_multi=args.pssm_multi, pssm_log_odds_flag=bool(args.pssm_log_odds_flag), pssm_log_odds_mask=pssm_log_odds_mask_caa, pssm_bias_flag=bool(args.pssm_bias_flag), bias_by_res=bias_by_res_all_caa)
                    sample_dict9 = model_9.sample(X_caa, randn_2_caa, S_caa, chain_M_caa, chain_encoding_all_caa, residue_idx_caa, mask=mask_caa, temperature=temp, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos_caa, omit_AA_mask=omit_AA_mask_caa, pssm_coef=pssm_coef_caa, pssm_bias=pssm_bias_caa, pssm_multi=args.pssm_multi, pssm_log_odds_flag=bool(args.pssm_log_odds_flag), pssm_log_odds_mask=pssm_log_odds_mask_caa, pssm_bias_flag=bool(args.pssm_bias_flag), bias_by_res=bias_by_res_all_caa)
                    
                    S_sample1 = sample_dict1["S"] 
                    S_sample2 = sample_dict2["S"] 
                    S_sample3 = sample_dict3["S"] 
                    S_sample4 = sample_dict4["S"] 
                    S_sample5 = sample_dict5["S"] 
                    S_sample6 = sample_dict6["S"] 
                    S_sample7 = sample_dict7["S"] 
                    S_sample8 = sample_dict8["S"] 
                    S_sample9 = sample_dict9["S"] 

                

                    _, mpnn_emb21 = model_1(X, S_sample1, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_2, use_input_decoding_order=True, decoding_order=sample_dict1["decoding_order"])
                    _, mpnn_emb22 = model_2(X, S_sample2, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_2, use_input_decoding_order=True, decoding_order=sample_dict2["decoding_order"])
                    _, mpnn_emb23 = model_3(X, S_sample3, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_2, use_input_decoding_order=True, decoding_order=sample_dict3["decoding_order"])
                    _, mpnn_emb24 = model_4(X, S_sample4, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_2, use_input_decoding_order=True, decoding_order=sample_dict4["decoding_order"])
                    _, mpnn_emb25 = model_5(X, S_sample5, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_2, use_input_decoding_order=True, decoding_order=sample_dict5["decoding_order"])
                    _, mpnn_emb26 = model_6(X, S_sample6, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_2, use_input_decoding_order=True, decoding_order=sample_dict6["decoding_order"])
                    _, mpnn_emb27 = model_7(X_caa, S_sample7, mask_caa, chain_M_caa*chain_M_pos_caa, residue_idx_caa, chain_encoding_all_caa, randn_2_caa, use_input_decoding_order=True, decoding_order=sample_dict7["decoding_order"])
                    _, mpnn_emb28 = model_8(X_caa, S_sample8, mask_caa, chain_M_caa*chain_M_pos_caa, residue_idx_caa, chain_encoding_all_caa, randn_2_caa, use_input_decoding_order=True, decoding_order=sample_dict8["decoding_order"])
                    _, mpnn_emb29 = model_9(X_caa, S_sample9, mask_caa, chain_M_caa*chain_M_pos_caa, residue_idx_caa, chain_encoding_all_caa, randn_2_caa, use_input_decoding_order=True, decoding_order=sample_dict9["decoding_order"])

                    mpnn_emb2 = torch.cat((
                        mpnn_emb21,
                        mpnn_emb22,
                        mpnn_emb23,
                        mpnn_emb24,
                        mpnn_emb25,
                        mpnn_emb26,
                        mpnn_emb27,
                        mpnn_emb28,
                        mpnn_emb29,
                    ),dim=-1)



                
            data_dict = {
            "coords": X.squeeze().detach(),
            "seq": ori_seq,
            "mask": mask.detach(),
            "mpnn_emb": mpnn_emb2.squeeze().detach(),
            "mpnn_emb_sup": mpnn_emb1.squeeze().detach()
            }

            filename = '/root/mpnnplus_cath/test/' + str(count) + '.pkl'
            save_dict_to_pickle(data_dict, filename)
            
            if count%100 == 0:
                print(count)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--suppress_print", type=int, default=0, help="0 for False, 1 for True")

  
    argparser.add_argument("--ca_only", action="store_true", default=False, help="Parse CA-only structures and use CA-only models (default: false)")   
    argparser.add_argument("--path_to_model_weights", type=str, default="", help="Path to model weights folder;") 
    argparser.add_argument("--model_name", type=str, default="v_48_020", help="ProteinMPNN model name: v_48_002, v_48_010, v_48_020, v_48_030; v_48_010=version with 48 edges 0.10A noise")
    argparser.add_argument("--use_soluble_model", action="store_true", default=False, help="Flag to load ProteinMPNN weights trained on soluble proteins only.")


    argparser.add_argument("--seed", type=int, default=37, help="If set to 0 then a random seed will be picked;")
 
    argparser.add_argument("--save_score", type=int, default=0, help="0 for False, 1 for True; save score=-log_prob to npy files")
    argparser.add_argument("--save_probs", type=int, default=0, help="0 for False, 1 for True; save MPNN predicted probabilites per position")

    argparser.add_argument("--score_only", type=int, default=0, help="0 for False, 1 for True; score input backbone-sequence pairs")
    argparser.add_argument("--path_to_fasta", type=str, default="", help="score provided input sequence in a fasta format; e.g. GGGGGG/PPPPS/WWW for chains A, B, C sorted alphabetically and separated by /")


    argparser.add_argument("--conditional_probs_only", type=int, default=0, help="0 for False, 1 for True; output conditional probabilities p(s_i given the rest of the sequence and backbone)")    
    argparser.add_argument("--conditional_probs_only_backbone", type=int, default=0, help="0 for False, 1 for True; if true output conditional probabilities p(s_i given backbone)") 
    argparser.add_argument("--unconditional_probs_only", type=int, default=0, help="0 for False, 1 for True; output unconditional probabilities p(s_i given backbone) in one forward pass")   
 
    argparser.add_argument("--backbone_noise", type=float, default=0.00, help="Standard deviation of Gaussian noise to add to backbone atoms")
    argparser.add_argument("--num_seq_per_target", type=int, default=1, help="Number of sequences to generate per target")
    argparser.add_argument("--batch_size", type=int, default=1, help="Batch size; can set higher for titan, quadro GPUs, reduce this if running out of GPU memory")
    argparser.add_argument("--max_length", type=int, default=200000, help="Max sequence length")
    argparser.add_argument("--sampling_temp", type=str, default="0.1", help="A string of temperatures, 0.2 0.25 0.5. Sampling temperature for amino acids. Suggested values 0.1, 0.15, 0.2, 0.25, 0.3. Higher values will lead to more diversity.")
    
    argparser.add_argument("--out_folder", type=str, default='/root/mpnn_cath/', help="Path to a folder to output sequences, e.g. /home/out/")
    argparser.add_argument("--pdb_path", type=str, default='', help="Path to a single PDB to be designed")
    argparser.add_argument("--pdb_path_chains", type=str, default='', help="Define which chains need to be designed for a single PDB ")
    argparser.add_argument("--jsonl_path", type=str, default='/root/chain_set_test.jsonl', help="Path to a folder with parsed pdb into jsonl")
    argparser.add_argument("--chain_id_jsonl",type=str, default='', help="Path to a dictionary specifying which chains need to be designed and which ones are fixed, if not specied all chains will be designed.")
    argparser.add_argument("--fixed_positions_jsonl", type=str, default='', help="Path to a dictionary with fixed positions")
    argparser.add_argument("--omit_AAs", type=list, default='X', help="Specify which amino acids should be omitted in the generated sequence, e.g. 'AC' would omit alanine and cystine.")
    argparser.add_argument("--bias_AA_jsonl", type=str, default='', help="Path to a dictionary which specifies AA composion bias if neededi, e.g. {A: -1.1, F: 0.7} would make A less likely and F more likely.")
   
    argparser.add_argument("--bias_by_res_jsonl", default='', help="Path to dictionary with per position bias.") 
    argparser.add_argument("--omit_AA_jsonl", type=str, default='', help="Path to a dictionary which specifies which amino acids need to be omited from design at specific chain indices")
    argparser.add_argument("--pssm_jsonl", type=str, default='', help="Path to a dictionary with pssm")
    argparser.add_argument("--pssm_multi", type=float, default=0.0, help="A value between [0.0, 1.0], 0.0 means do not use pssm, 1.0 ignore MPNN predictions")
    argparser.add_argument("--pssm_threshold", type=float, default=0.0, help="A value between -inf + inf to restric per position AAs")
    argparser.add_argument("--pssm_log_odds_flag", type=int, default=0, help="0 for False, 1 for True")
    argparser.add_argument("--pssm_bias_flag", type=int, default=0, help="0 for False, 1 for True")
    
    argparser.add_argument("--tied_positions_jsonl", type=str, default='', help="Path to a dictionary with tied positions")
    
    args = argparser.parse_args()    
    main(args)   

