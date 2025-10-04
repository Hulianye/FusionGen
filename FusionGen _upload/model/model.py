import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from typing import Optional

class FusionGen(nn.Module):
    def __init__(self, channel=15):
        super(FusionGen, self).__init__()

        # Encoder (All convolution kernels have height=1 to maintain 22-dimensional spatial characteristics)
        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 25), stride=(1, 5), padding=(0, 12)),
            nn.LayerNorm([8, channel, 200]),
        )

        self.encoder_conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(1, 15), stride=(1, 5), padding=(0, 7)),
            nn.LayerNorm([16, channel, 40]),
        )

        self.encoder_conv3 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=(1, 15), stride=(1, 2), padding=(0, 7)),
            nn.LayerNorm([8, channel, 20]),
        )

        # Decoder (Transposed convolution to restore temporal dimension)
        self.decoder_deconv1 = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=(1, 15), stride=(1, 2),
                               padding=(0, 7), output_padding=(0, 1)),
            nn.LayerNorm([16, channel, 40]),
        )

        self.decoder_conv1 = nn.Conv2d(32, 16, kernel_size=3, padding=1)

        self.decoder_deconv2 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=(1, 15), stride=(1, 5),
                               padding=(0, 7), output_padding=(0, 4)),
            nn.LayerNorm([8, channel, 200]),
        )

        self.decoder_conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)

        self.decoder_deconv3 = nn.Sequential(
            nn.ConvTranspose2d(8, 1, kernel_size=(1, 25), stride=(1, 5),
                               padding=(0, 12), output_padding=(0, 4)),
        )

        self.ref_features = None
        self.ref_labels = None

    def forward(self, x):
        # Encoding phase
        x = x.unsqueeze(dim=1)
        enc1 = self.encoder_conv1(x)  # (B,8,200)
        enc2 = self.encoder_conv2(enc1)  # (B,16,40)
        enc3 = self.encoder_conv3(enc2)  # (B,4,20)

        # Decoding phase with skip connections
        dec1 = self.decoder_deconv1(enc3)  # (B,16,40)
        dec1 = torch.cat([dec1, enc2], dim=1)  # Channel concatenation
        dec1 = self.decoder_conv1(dec1)

        dec2 = self.decoder_deconv2(dec1)  # (B,8,200)
        dec2 = torch.cat([dec2, enc1], dim=1)
        dec2 = self.decoder_conv2(dec2)

        out = self.decoder_deconv3(dec2)  # (B,22,1000)
        out = out.squeeze(dim=1)
        return out

    def encode_reference(self, x_ref, labels_ref):
        """Pre-encode reference data"""
        with torch.no_grad():
            # Execute complete encoding process
            x_ref = x_ref.unsqueeze(1)
            enc1_ref = self.encoder_conv1(x_ref)
            enc2_ref = self.encoder_conv2(enc1_ref)
            enc3_ref = self.encoder_conv3(enc2_ref)

            # Store features from each layer
            self.ref_features = {
                'enc1': enc1_ref,
                'enc2': enc2_ref,
                'enc3': enc3_ref,
                'data': x_ref
            }
            self.ref_labels = labels_ref

    def _find_ref_peers(self, current_label, enc3):
        """Randomly extract a sample of the same class from reference data"""
        if self.ref_labels is None:
            return None

        # Get all indices of samples with the same label
        same_mask = self.ref_labels == current_label
        peers = torch.where(same_mask)[0]

        # Return empty if no samples of the same class
        if not peers.numel():
            return None

        # Randomly select an index (maintain tensor dimension)
        rand_idx = torch.randint(low=0, high=len(peers), size=(1,))
        return peers[rand_idx]

    def _fuse_with_ref(self, enc3, labels, alpha=0.2):
        """Feature fusion using combined spatiotemporal dimensions"""
        B, C, H, T = enc3.shape
        total_positions = H * T
        n_select = int(total_positions * alpha)

        fused_enc3 = enc3.clone()
        replacement_map = {}

        # Flatten current features and reference features
        flat_enc3 = enc3.view(B, C, -1)  # [B, C, H*T]
        flat_ref = self.ref_features['enc3'].view(len(self.ref_labels), C, -1)  # [n_ref, C, H*T]

        result_env = enc3.clone()

        for i in range(B):
            peers_ref = self._find_ref_peers(labels[i].item(), enc3[i])
            if not peers_ref.numel():
                continue

            # Randomly select position indices
            selected_pos = torch.randperm(total_positions)[:n_select]

            # Convert selected positions of current sample to coordinates
            h_current = (selected_pos // T).tolist()
            t_current = (selected_pos % T).tolist()

            # Get query features [n_select, C]
            query_feats = flat_enc3[i, :, selected_pos].permute(1, 0)  # [n_select, C]

            # Get reference features [n_peer, H*T, C]
            ref_feats = flat_ref[peers_ref].permute(0, 2, 1)  # [n_peer, H*T, C]

            # Calculate similarity matrix [n_select, n_peer*H*T]
            sim_matrix = F.cosine_similarity(
                query_feats.unsqueeze(1).unsqueeze(2),  # [n_select, 1, 1, C]
                ref_feats.unsqueeze(0),  # [1, n_peer, H*T, C]
                dim=-1
            ).view(len(query_feats), -1)

            # Find best matches
            max_sim, max_indices = torch.max(sim_matrix, dim=1)
            peer_indices = max_indices // (H * T)  # Reference sample indices
            ref_pos = max_indices % (H * T)  # Reference sample position indices

            # Convert reference sample positions to coordinates
            ref_h = (ref_pos // T).tolist()
            ref_t = (ref_pos % T).tolist()

            # Record replacement information
            replacement_map[i] = {
                'h_current': h_current,
                't_current': t_current,
                'peer_indices': peers_ref[peer_indices].tolist(),
                'ref_h': ref_h,
                'ref_t': ref_t
            }
            result_env[i] = self.ref_features['enc3'][peers_ref[peer_indices[0]]]
            
            # Perform feature replacement (using current sample's position)
            for idx in range(n_select):
                h_cur = h_current[idx]
                t_cur = t_current[idx]
                peer_idx = peers_ref[peer_indices[idx]]
                h_ref = ref_h[idx]
                t_ref = ref_t[idx]

                result_env[i, :, h_ref, t_ref] = fused_enc3[i, :, h_cur, t_cur]

        return result_env, replacement_map

    def _propagate_ref_replacements(self, layer_feat, replacement_map, ref_layer_name, time_scale):
        """Improved feature propagation method based on correct positions"""
        fused_feat = layer_feat.clone()
        ref_feat = self.ref_features[ref_layer_name]
        _, _, H, T = layer_feat.shape

        for i in replacement_map:
            info = replacement_map[i]
            fused_feat[i] = ref_feat[info['peer_indices'][0]]
            for h_cur, t_cur, peer_idx, h_ref, t_ref in zip(info['h_current'], info['t_current'],
                                                            info['peer_indices'], info['ref_h'], info['ref_t']):
                # Calculate propagation range for current sample
                current_start = t_cur * time_scale
                current_end = current_start + time_scale

                # Calculate propagation range for reference sample
                ref_start = t_ref * time_scale
                ref_end = ref_start + time_scale

                # Boundary checks
                current_end = min(current_end, T)
                ref_end = min(ref_end, ref_feat.shape[3])

                # Perform weighted fusion
                fused_feat[i, :, h_ref, ref_start:ref_end] = layer_feat[i, :, h_cur, current_start:current_end]

        return fused_feat

    def fusedforward(self, x, labels, alpha=0.9):
        x = x.unsqueeze(1)

        # Encoding phase
        enc1 = self.encoder_conv1(x)
        enc2 = self.encoder_conv2(enc1)
        enc3 = self.encoder_conv3(enc2)

        # Feature fusion using reference data
        enc3_fused, rep_map = self._fuse_with_ref(enc3, labels, alpha)

        # Propagate replacements to other layers
        enc2_fused = self._propagate_ref_replacements(enc2, rep_map, 'enc2', 2)
        enc1_fused = self._propagate_ref_replacements(enc1, rep_map, 'enc1', 10)

        # Decoding phase
        dec1 = self.decoder_deconv1(enc3_fused)
        dec1 = torch.cat([dec1, enc2_fused], dim=1)
        dec1 = self.decoder_conv1(dec1)

        dec2 = self.decoder_deconv2(dec1)
        dec2 = torch.cat([dec2, enc1_fused], dim=1)
        dec2 = self.decoder_conv2(dec2)

        return self.decoder_deconv3(dec2).squeeze(1)