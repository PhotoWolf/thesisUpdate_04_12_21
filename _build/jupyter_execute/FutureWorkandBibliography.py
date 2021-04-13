# Future Directions

- Thus far, we have only explored centrality measures that can be easily expressed as the weighted sum of matrix powers. Path-based centrality (i.e betweeness, closeness, etc.) may pose additional problems for standard GCN architectures. Moreover, while even Katz Centrality is fairly quick to compute for small to medium sized networks, path centralities do not scale as well. The fastest algorithm is that of Brandes [17], which is $O(|N||E|)$. It is not as trivial to merely precompute these metrics and append them as node features, so a network capable of learning a decent approximation is of value.
- Regarding our current work, there are a few avenues we'd like to explore further. Overmsoothing and training deeper GCNs is an active area of study in the community, and the theoretical side is not all that well developed. We are vaguely aware of a work linking oversmoothing to a matrix's spectral gap in the context of expander graphs, but nothing in terms of eigenvalue ratio, dominant eigenvectors, etc. There are also some custom architectures we'd like to try, in so far as they might perform better than GraphConv. The code for these is already finished (I spent Janurary playing around with a few different ideas); it is just a matter of testing them on our synthetic dataset. 

# Bibliography

1. Gilmer, J., Schoenholz, S., Riley, P.F., Vinyals, O., & Dahl, G.E. (2017). Neural Message Passing for Quantum Chemistry. ArXiv, abs/1704.01212.
2. Hu, Z., Fan, C., Chen, T., Chang, K., & Sun, Y. (2019). Pre-Training Graph Neural Networks for Generic Structural Feature Extraction. ArXiv, abs/1905.13728.
3. Dwivedi, V.P., Joshi, C.K., Laurent, T., Bengio, Y., & Bresson, X. (2020). Benchmarking Graph Neural Networks. ArXiv, abs/2003.00982.
4. Chen, Hongxu & Yin, Hongzhi & Chen, Tong & Hung, Nguyen & Peng, Wen-Chih & Li, Xue. (2019). Exploiting Centrality Information with Graph Convolutions for Network Representation Learning. 590-601. 10.1109/ICDE.2019.00059. 
5. Mendonça, M.R., Barreto, A., & Ziviani, A. (2021). Approximating Network Centrality Measures Using Node Embedding and Machine Learning. IEEE Transactions on Network Science and Engineering, 8, 220-230.
6. Grando, F., Granville, L., & Lamb, L. (2019). Machine Learning in Network Centrality Measures. ACM Computing Surveys (CSUR), 51, 1 - 32.
7. Morris, C., Ritzert, M., Fey, M., Hamilton, W.L., Lenssen, J.E., Rattan, G., & Grohe, M. (2019). Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks. AAAI.
8. Velickovic, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph Attention Networks. ArXiv, abs/1710.10903.
9. Wang, Y., Sun, Y., Liu, Z., Sarma, S.E., Bronstein, M., & Solomon, J. (2019). Dynamic Graph CNN for Learning on Point Clouds. ACM Transactions on Graphics (TOG), 38, 1 - 12.
10. Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. ArXiv, abs/1502.03167.
11. Maas, A.L. (2013). Rectifier Nonlinearities Improve Neural Network Acoustic Models.
12. Karrer, B., & Newman, M. (2011). Stochastic blockmodels and community structure in networks. Physical review. E, Statistical, nonlinear, and soft matter physics, 83 1 Pt 2, 016107 .
13. Katz, L. A new status index derived from sociometric analysis. Psychometrika 18, 39–43 (1953).
14. Zhao, L., & Akoglu, L. (2020). PairNorm: Tackling Oversmoothing in GNNs. ArXiv, abs/1909.12223.
15. Wu, F., Zhang, T., Souza, A., Fifty, C., Yu, T., & Weinberger, K.Q. (2019). Simplifying Graph Convolutional Networks. ArXiv, abs/1902.07153.
16. Abu-El-Haija, S., Perozzi, B., Kapoor, A., Harutyunyan, H., Alipourfard, N., Lerman, K., Steeg, G.V., & Galstyan, A. (2019). MixHop: Higher-Order Graph Convolutional Architectures via Sparsified Neighborhood Mixing. ICML.
17. Brandes, U. (2001). A faster algorithm for betweenness centrality. The Journal of Mathematical Sociology, 25, 163 - 177.