#----------------------------------------------------------------------------
# Non-saturating logistic loss with path length regularizer from the paper
# "Analyzing and Improving the Image Quality of StyleGAN", Karras et al. 2019

def G_logistic_ns_pathreg(G, D, opt, training_set, minibatch_size, pl_minibatch_shrink=2, pl_decay=0.01, pl_weight=2.0):
    _ = opt
    # random normal latent input
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    # random labels
    labels = training_set.get_random_labels_tf(minibatch_size)

    # generate images, latent space (w from W)
    fake_images_out, fake_dlatents_out = G.get_output_for(latents, labels, is_training=True, return_dlatents=True)

    # this is just for discriminator; not needed for path length regularization?
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    loss = tf.nn.softplus(-fake_scores_out) # -log(sigmoid(fake_scores_out))


    # fake_images_out   -> generated images
    # fake_dlatents_out -> intermediate latents space (w)

    # generate single (summed) change to the images
    random_normal_noise = tf.random.normal(tf.shape(fake_images_out)) / image_width
    ys = tf.math.reduce_sum(fake_images_out * random_normal_noise)
    # generate gradients -> for each x in fake_dlatents_out, get d(image)/d(x)
    # now we have a bunch of gradients; one for each element in fake_dlatents_out,
    # which includes the batch dimension and any other dimensions. In our model,
    # this will have shape [N,w], where N is the batch size, and w is the size of the latent space
    grads = tf.gradients(ys, [fake_dlatents_out])[0]

    # Path length regularization.
    with tf.name_scope('PathReg'):

        # Evaluate the regularization term using a smaller minibatch to conserve memory.
        #if pl_minibatch_shrink > 1:
        #    pl_minibatch = minibatch_size // pl_minibatch_shrink
        #    # set new latents to normally distributed random numbers
        #    pl_latents = tf.random_normal([pl_minibatch] + G.input_shapes[0][1:])
        #    # set new random labels
        #    pl_labels = training_set.get_random_labels_tf(pl_minibatch)
        #    # get fake images (from pl_latents) and fake latent space (from pl_labels)
        #    fake_images_out, fake_dlatents_out = G.get_output_for(pl_latents, pl_labels, is_training=True, return_dlatents=True)

        # Compute |J*y|.
        # generate noise; I believe this is used to 'perturb' the fake images?? np.sqrt(np.prod(dims)) makes this normalized
        pl_noise = tf.random_normal(tf.shape(fake_images_out)) / np.sqrt(np.prod(G.output_shape[2:]))
        # 1. multiply fake_images_out * pl_noise -> perturb fake images in preparation for gradient calculations?
        # 2. reduce_sum the perturbed fake images -> this generates ys (one for each batch) for gradient calculation?
        # 3. calculate gradient of ys wrt fake_dlatents_out (what is this??, maybe just original latent space); only keep the first one (I think this just removes the 'list' part of the return value)
        # now pl_grads should be a single number per batch element, d(g(w) . y) wrt w
        # pl_grads estimates JTwy.
        pl_grads = tf.gradients(tf.reduce_sum(fake_images_out * pl_noise), [fake_dlatents_out])[0]
        # Now calculate the 2-norm (euclidean norm) of pl_grads (estimate of JTwy)
        # 1. square gradients element-wise (gradients are 1D 'distances')
        # 2. reduce_sum of squares (axis=2?? 0->batches, 1->layers(gen_blocks??), 2->channels?)
        # 3. reduce_mean (axis=1?? I think this might be averaging over all blocks in the generator)
        # 4. take sqrt -> now we have 2-norm in pl_lengths, which should now be a single number!!
        pl_lengths = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(pl_grads), axis=2), axis=1))
        # autosummary is just for tracking (it's a no-op)
        pl_lengths = autosummary('Loss/pl_lengths', pl_lengths)

        # Track exponential moving average of |J*y|.
        with tf.control_dependencies(None):
            pl_mean_var = tf.Variable(name='pl_mean', trainable=False, initial_value=0.0, dtype=tf.float32)
        # this just averages all pl_lengths over batches?
        pl_mean = pl_mean_var + pl_decay * (tf.reduce_mean(pl_lengths) - pl_mean_var)
        pl_update = tf.assign(pl_mean_var, pl_mean)

        # Calculate (|J*y|-a)^2.
        with tf.control_dependencies([pl_update]):
            # we have ||JTwy||2 (aka "|J*y|") as a single numbe rin pl_lengths
            # subtract the running mean ("a"), and square it to get the pl_penalty
            pl_penalty = tf.square(pl_lengths - pl_mean)
            pl_penalty = autosummary('Loss/pl_penalty', pl_penalty)

        # Apply weight.
        #
        # Note: The division in pl_noise decreases the weight by num_pixels, and the reduce_mean
        # in pl_lengths decreases it by num_affine_layers. The effective weight then becomes:
        #
        # gamma_pl = pl_weight / num_pixels / num_affine_layers
        # = 2 / (r^2) / (log2(r) * 2 - 2)
        # = 1 / (r^2 * (log2(r) - 1))
        # = ln(2) / (r^2 * (ln(r) - ln(2))
        #
        reg = pl_penalty * pl_weight

    return loss, reg

#----------------------------------------------------------------------------
