# What the foundation-model checkpoints were pre-trained on

This page says what data stands behind the large models we submitted. Anyone using this project to judge "the limit of EEG foundation models" should know what the foundation model had seen.

## The final packages

Every foundation-model member of every package we uploaded in the sealed phase starts from the same published checkpoint. The members of the final entry (0.844) were built in three steps; earlier packages used the same starting checkpoint with an earlier version of step 2 on the same three public sources.

1. **Start: the published ST-EEGFormer-large checkpoint.** I am one of the authors of that model.
2. **Continued pre-training by us** (masked-autoencoder objective, no labels) on windows of three public sources: the released competition sessions (PROTEUS), Dreyer 2023 and Scherer 2015. One of the six members instead received a supervised pre-fine-tune on public motor-imagery datasets.
3. **Fine-tuning by us** on the labelled competition sessions that the organisers released.

Our own training runs for this entry (steps 2 and 3) read public data only.

## What the starting checkpoint itself was pre-trained on

As one of its authors I can say what its pre-training corpus includes:

- public BCI datasets, among them a split of BCI Competition IV-2a;
- the SEED series, which is available under a licence agreement with its providers and is not freely redistributable;
- in-house lab recordings that have not been released.

We did not access any of these recordings in this project; we started from the published weights.

## An earlier upload that used a second in-house model

One of the three packages uploaded on the first sealed day (Nov 5; 0.798 after the organisers' re-run; never our best entry) also contained MOR, a new in-house foundation model that is still in development. It is unpublished, and its pre-training data is not described here. MOR was not used in any later Track 2 package.

## How this relates to the competition's rules

The competition's Rule 4 allows pre-training on public datasets whose terms permit it, asks that every external corpus be declared, and states that non-permitted closed clinical datasets are not allowed. It does not say how a published checkpoint with partly non-public pre-training data is to be treated, and several widely used published EEG foundation models are in that position.

We have no ruling on this point. Our final entry finished outside the top three and was not part of the reproducibility audit.

I state this here so that the result is read for what it is: a system fine-tuned from a published checkpoint whose own pre-training was not limited to freely redistributable data.

## The one test of a "clean" foundation model

To see what was lost without the published checkpoint, a smaller ST-EEGFormer (base size) was pre-trained from scratch on freely redistributable data only, for about 19 GPU-hours. As an ensemble member it was 0.12 below the member built on the published large checkpoint, and the package containing it was no better than the package with no foundation model at all (T2-D356). The comparison is uneven (smaller model, far less pre-training), so it shows only that a quick clean replacement was not available to us, not what a properly resourced one could do.
