# class OamACE(Product):
#     def _setup(self):

#         self.linear_up = Linear(
#             self.irreps_in,
#             self.irreps_hidden1,
#             bias=self.use_bias,
#         ) if self.num_channel != self.num_hidden_channel else torch.nn.Identity()

#         self.ace = uuuTensorProduct(
#             irreps_in1=self.irreps_hidden1,
#             irreps_in2=self.irreps_hidden1[:1] + self.irreps_hidden1,
#             irreps_out=self.irreps_hidden2,
#             l1l2=self.l1l2,
#             l3s=self.l3s,
#             trainable=True,
#         ) 
#         self.coef = torch.nn.Parameter(torch.randn(self.num_elements, self.ace.weight_numel))

#         irreps_gated = self.irreps_hidden2
#         irreps_gates = o3.Irreps([mul, (0, 1)] for mul, _ in self.irreps_hidden2)
#         self.nonlinearity = GatedLinearUnit(
#             irreps_gates=irreps_gates,
#             act_gates=[ACTIVATION[self.nonlinear_act]()] * len(irreps_gates),
#             irreps_gated=irreps_gated,
#         )

#         self.linear_gate = Linear(
#             self.ace.irreps_out.simplify(),
#             self.nonlinearity.irreps_in,
#             bias=self.use_bias,
#         )

#         self.linear_down = Linear(
#             self.irreps_hidden2,
#             self.irreps_out,
#             bias=self.use_bias
#         )    
        
#     def forward(
#             self, 
#             node_feats: torch.Tensor, 
#             node_attrs: torch.Tensor,
#             sc: torch.Tensor,
#             batch: torch.Tensor,
#         ) -> torch.Tensor:

#         node_feats = self.linear_up(node_feats)
#         ones = node_feats.new_ones(node_feats.size(0), self.num_hidden_channel)

#         corr_feats = self.ace(
#             node_feats, 
#             torch.cat(
#                 [
#                     ones,
#                     node_feats,
#                 ],
#                 dim=-1,
#             ), 
#             torch.einsum('bz, zi -> bi', node_attrs, self.coef),
#         )

#         outs = self.linear_down(self.nonlinearity(self.linear_gate(corr_feats)))

#         if sc is not None:
#             outs = outs + sc

#         return outs
    



