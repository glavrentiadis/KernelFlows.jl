This is a package to construct cross-validated feature maps and Gaussian Proces models for for moderate-to high-dimensional inference. It is witten in the Julia programming language and has been used for a number of studies, including the following:

- Di Natale, C. M., Schaber, T., Tran, G., Susiluoto, J., Keller, D., Ekholm, T., & Partanen, A.-I. (2025): *The coupled uncertainties in carbon dioxide removal and transient climate response to cumulative CO₂ emissions*. **Environmental Research Letters**. https://doi.org/10.1088/1748-9326/ae20a5

- Lamminpää, O., Susiluoto, J., Hobbs, J., McDuffie, J., Braverman, A., and Owhadi, H. (2025): *Forward model emulator for atmospheric radiative transfer using Gaussian processes and cross validation*, **Atmos. Meas. Tech.**, 18, 673–694, https://doi.org/10.5194/amt-18-673-2025

For general usage, see example\_script.jl. For GPU-assisted updating of matrices with large training data volumes, see scripts in the GPU\_scripts directory. A publication to describe the technology in detail is currently in prep.
