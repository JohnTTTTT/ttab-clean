from ttab.scenarios import Scenario, TestDomain, TestCase  # scenario API imports ([github.com](https://github.com/LINs-lab/ttab))
from ttab.scenarios import HomogeneousNoMixture           # inter-domain mixing control ([github.com](https://github.com/LINs-lab/ttab))

# Define default scenarios dictionary for exps scripts
default_scenarios = {
    "affectnet_to_rafdb_natural": Scenario(
        task="classification",
        model_name="vit_large_patch16_224",
        model_adaptation_method="tent",            # use TENT for test-time adaptation ([github.com](https://github.com/LINs-lab/ttab))
        model_selection_method="last_iterate",      # select final iterate as output ([github.com](https://github.com/LINs-lab/ttab))
        
        base_data_name="affectnet",                # source dataset identifier ([arxiv.org](https://arxiv.org/pdf/1708.03985?utm_source=chatgpt.com))
        test_domains=[
            TestDomain(
                base_data_name="affectnet",
                data_name="rafdb",                  # target dataset identifier ([paperswithcode.com](https://paperswithcode.com/dataset/raf-db?utm_source=chatgpt.com), [kaggle.com](https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset?utm_source=chatgpt.com))
                shift_type="natural",               # natural covariate shift ([github.com](https://github.com/LINs-lab/ttab))
                domain_sampling_name="uniform",     # uniform domain sampling ([github.com](https://github.com/LINs-lab/ttab))
                domain_sampling_value=None,
                domain_sampling_ratio=1.0,
            )
        ],
        test_case=TestCase(
            inter_domain=HomogeneousNoMixture(has_mixture=False),
            batch_size=16,                         # per-batch adaptation size ([github.com](https://github.com/LINs-lab/ttab))
            data_wise="batch_wise",             # batch-wise adaptation ([github.com](https://github.com/LINs-lab/ttab))
            offline_pre_adapt=False,
            episodic=False,
            intra_domain_shuffle=True,
        ),
    )
}
