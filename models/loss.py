import tensorflow as tf

def Y_to_labels(Y):  #  Y  (25461,1)
    labels = tf.equal(Y, tf.transpose(Y))
    labels = tf.cast(labels, tf.float32)
    label_upper = (tf.linalg.band_part(labels, 0, -1) - tf.linalg.band_part(labels, 0, 0))[:, 1:]
    label_lower = (tf.linalg.band_part(labels, -1, 0) - tf.linalg.band_part(labels, 0, 0))[:, :-1]
    labels = label_upper + label_lower
    return labels

def Z_to_logits(Z, tao):
    tao = tf.cast(tao, tf.float32)
    sim = tf.matmul(Z, Z, transpose_b=True)/tao      # BxC x CxB = BxB
    mask_upper = (tf.linalg.band_part(sim, 0, -1) - tf.linalg.band_part(sim, 0, 0))[:, 1:]
    mask_lower = (tf.linalg.band_part(sim, -1, 0) - tf.linalg.band_part(sim, 0, 0))[:, :-1]
    logits = mask_upper + mask_lower
    return logits

def compute_iobs(word, summ, sort, num):  #
    summ = tf.cast(summ, tf.float32)
    word = tf.cast(word, tf.float32)
    sort = tf.cast(sort, tf.float32)
    # compute |sort[i, :] - sort[j, :]|
    differ = tf.abs(tf.expand_dims(sort, axis=1) - tf.expand_dims(sort, axis=0))  # (N, 1, D) - (1, N, D) =  (N, N, D)
    # compute (entro_word[i, :] + entro_word[j, :])
    word = tf.expand_dims(word, axis=1) + tf.expand_dims(word, axis=0)  # (N, 1, D) +  (1, N, D) = (N, N, D)
    # compute (entro_sum[i, 0] + entro_sum[j, 0])
    summ = tf.expand_dims(summ[:, 0], axis=1) + tf.expand_dims(summ[:, 0], axis=0)  # (N, 1) + (1, N) = (N, N)
    differ = differ * word / tf.expand_dims(summ, axis=2)  # (N, N, D)
    prior_sim = tf.reduce_sum(differ, axis=2) / num    # (N, N)   # sim

    prior_sim = tf.cast(prior_sim, tf.float32)
    label_upper = (tf.linalg.band_part(prior_sim, 0, -1) - tf.linalg.band_part(prior_sim, 0, 0))[:, 1:]
    label_lower = (tf.linalg.band_part(prior_sim, -1, 0) - tf.linalg.band_part(prior_sim, 0, 0))[:, :-1]
    prior_sim = label_upper + label_lower

    x_min = tf.reduce_min(prior_sim)  # 0
    x_max = tf.reduce_max(prior_sim)  # 0.539410412
    prior_sim = 1.0 - (prior_sim - x_min) / (x_max - x_min)
    return prior_sim


def compute_pearson(X):
    mean_X = tf.reduce_mean(X, axis=1, keepdims=True)
    X_centered = X - mean_X
    numerator = tf.matmul(X_centered, X_centered, transpose_b=True) / 250.0
    std_X = tf.math.reduce_std(X, axis=1, keepdims=True)
    denominator = tf.matmul(std_X, std_X, transpose_b=True)
    prior_sim = numerator / denominator

    # tf.print("mean_X", mean_X)
    # tf.print("std_X", std_X)
    # tf.print("prior_sim", prior_sim)
    label_upper = (tf.linalg.band_part(prior_sim, 0, -1) - tf.linalg.band_part(prior_sim, 0, 0))[:, 1:]
    label_lower = (tf.linalg.band_part(prior_sim, -1, 0) - tf.linalg.band_part(prior_sim, 0, 0))[:, :-1]
    prior_sim = label_upper + label_lower

    # pearson (-1, 1)
    prior_sim = tf.abs(prior_sim)   # The closer to 0, the less similar.
    return prior_sim

def con_loss(logit,label):
    a = tf.exp(logit)
    n0 = tf.reduce_sum( tf.cast(label == 0, tf.float32), axis=1, keepdims=True)
    n1 = tf.reduce_sum(label, axis=1, keepdims=True)
    G = tf.reduce_sum(a * tf.cast(label == 0, tf.float32), axis=1, keepdims=True) / n0
    logit = a / (a + G)
    loss = tf.reduce_sum( -tf.math.log(logit) * label)/ n1
    return loss

def con_loss2(logit,label, weight):     # label  512*513
    a = tf.exp(logit)
    w = tf.exp(weight)
    n0 = tf.reduce_sum( tf.cast(label == 0, tf.float32), axis=1, keepdims=True)
    n1 = tf.reduce_sum( label, axis=1, keepdims=True)
    G = tf.reduce_sum(a * tf.cast(label == 0, tf.float32), axis=1, keepdims=True) / n0
    logit = a / (a + G) * w
    loss = tf.reduce_sum( -tf.math.log(logit) * label)/ n1
    return loss

# ------------------------------- all loss in here ----------------------------------------
def sup(c1, c0, e1, e0, a=0, num=24):
    e1 = tf.cast(e1, tf.float32)
    e0 = tf.cast(e0, tf.float32)
    def loss_fn(the_all, Z, tao=0.1):
        B = tf.shape(Z)[0]
        Y, entro_word, entro_sum, sort = tf.split(the_all, [1, num, 1, num], axis=-1)
        entro_word1, entro_sum1, sort1 = tf.split(e1, [num, 1, num], axis=-1)
        entro_word0, entro_sum0, sort0 = tf.split(e0, [num, 1, num], axis=-1)

        entro_word = tf.concat([entro_word, entro_word1, entro_word0], axis=0)
        entro_sum = tf.concat([entro_sum, entro_sum1, entro_sum0], axis=0)
        sort = tf.concat([sort, sort1, sort0], axis=0)  # (123,16) (1,16) concat

        all_prior_sim = compute_iobs(entro_word, entro_sum, sort, num)
        prior_sim = all_prior_sim[:B,:B-1]  # 128*127
        prior_sim1 = all_prior_sim[:B, B-1]
        prior_sim0 = all_prior_sim[:B, B]

        hard_labels = Y_to_labels(Y)
        logits = Z_to_logits(Z, tao)
        Y = tf.cast(Y, tf.float32)
        NY = tf.cast(1 - Y, tf.float32)
        weight = tf.zeros_like(logits)  #  128*129
        logits = logits + 0 * a * prior_sim * (1 - hard_labels) / tao
        prior_sim1 = tf.reshape(prior_sim1, (B, 1))
        prior_sim0 = tf.reshape(prior_sim0, (B, 1))
        sim1 = tf.matmul(Z, c1, ) / tao   # BxC x Cx1 = Bx1
        sim0 = tf.matmul(Z, c0, ) / tao
        all_logits = tf.concat([logits, sim1, sim0], axis=-1)
        w1 = - 0 * prior_sim1 * Y / tao
        w0 = - 0 * prior_sim0 * NY / tao

        weight = tf.concat([weight, w1, w0], axis=-1)
        all_hard_labels = tf.concat([hard_labels, Y, NY], axis=-1)

        loss = con_loss2(all_logits, all_hard_labels, weight)
        loss = loss / tf.cast(B, tf.float32)
        return loss
    return loss_fn


def sup_obs1(c1, c0, e1, e0, a,  num=24):
    e1 = tf.cast(e1, tf.float32)
    e0 = tf.cast(e0, tf.float32)
    def loss_fn(the_all, Z, tao=0.1):
        B = tf.shape(Z)[0]
        Y, entro_word, entro_sum, sort = tf.split(the_all, [1, num, 1, num], axis=-1)
        entro_word1, entro_sum1, sort1 = tf.split(e1, [num, 1, num], axis=-1)
        entro_word0, entro_sum0, sort0 = tf.split(e0, [num, 1, num], axis=-1)

        entro_word = tf.concat([entro_word, entro_word1, entro_word0], axis=0)
        entro_sum = tf.concat([entro_sum, entro_sum1, entro_sum0], axis=0)
        sort = tf.concat([sort, sort1, sort0], axis=0)  # (123,16) (1,16) concat

        all_prior_sim = compute_iobs(entro_word, entro_sum, sort, num)
        prior_sim = all_prior_sim[:B,:B-1]  # 128*127
        prior_sim1 = all_prior_sim[:B, B-1]
        prior_sim0 = all_prior_sim[:B, B]

        hard_labels = Y_to_labels(Y)
        logits = Z_to_logits(Z, tao)
        Y = tf.cast(Y, tf.float32)
        NY = tf.cast(1 - Y, tf.float32)
        weight = tf.zeros_like(logits)  # 128*129
        logits = logits + a * prior_sim * (1 - hard_labels) / tao
        prior_sim1 = tf.reshape(prior_sim1, (B, 1))
        prior_sim0 = tf.reshape(prior_sim0, (B, 1))
        sim1 = tf.matmul(Z, c1, ) / tao   # BxC x Cx1 = Bx1
        sim0 = tf.matmul(Z, c0, ) / tao
        all_logits = tf.concat([logits, sim1, sim0], axis=-1)
        w1 = - 0 * prior_sim1 * Y / tao
        w0 = - 0 * prior_sim0 * NY / tao

        weight = tf.concat([weight, w1, w0], axis=-1)
        all_hard_labels = tf.concat([hard_labels, Y, NY], axis=-1)

        loss = con_loss2(all_logits, all_hard_labels, weight)
        loss = loss / tf.cast(B, tf.float32)
        return loss
    return loss_fn



def sup_obs2(c1, c0, e1, e0, a,  num=24):
    e1 = tf.cast(e1, tf.float32)
    e0 = tf.cast(e0, tf.float32)
    def loss_fn(the_all, Z, tao=0.1):
        B = tf.shape(Z)[0]
        Y, entro_word, entro_sum, sort = tf.split(the_all, [1, num, 1, num], axis=-1)
        entro_word1, entro_sum1, sort1 = tf.split(e1, [num, 1, num], axis=-1)
        entro_word0, entro_sum0, sort0 = tf.split(e0, [num, 1, num], axis=-1)

        entro_word = tf.concat([entro_word, entro_word1, entro_word0], axis=0)
        entro_sum = tf.concat([entro_sum, entro_sum1, entro_sum0], axis=0)
        sort = tf.concat([sort, sort1, sort0], axis=0)  # (123,16) (1,16) concat

        all_prior_sim = compute_iobs(entro_word, entro_sum, sort, num)
        prior_sim = all_prior_sim[:B,:B-1]  # 128*127
        prior_sim1 = all_prior_sim[:B, B-1]
        prior_sim0 = all_prior_sim[:B, B]

        hard_labels = Y_to_labels(Y)
        logits = Z_to_logits(Z, tao)
        Y = tf.cast(Y, tf.float32)
        NY = tf.cast(1 - Y, tf.float32)
        weight = tf.zeros_like(logits)   #   128*129
        logits = logits + 0 * prior_sim * (1 - hard_labels) / tao
        prior_sim1 = tf.reshape(prior_sim1, (B, 1))
        prior_sim0 = tf.reshape(prior_sim0, (B, 1))
        sim1 = tf.matmul(Z, c1, ) / tao   # BxC x Cx1 = Bx1
        sim0 = tf.matmul(Z, c0, ) / tao
        all_logits = tf.concat([logits, sim1, sim0], axis=-1)
        w1 = - a * prior_sim1 * Y / tao
        w0 = - a * prior_sim0 * NY / tao

        weight = tf.concat([weight, w1, w0], axis=-1)
        all_hard_labels = tf.concat([hard_labels, Y, NY], axis=-1)

        loss = con_loss2(all_logits, all_hard_labels, weight)
        loss = loss / tf.cast(B, tf.float32)
        return loss
    return loss_fn


# c1 and c0 are the class centers for SA and NSA, respectively. e1 and e0 are the values for OBS.
# (Prior knowledge adjusts difficult negative sample pairs and simple positive sample pairs.)
def sup_obs(c1, c0, e1, e0, a,  num=24):  #  num=24 obs
    e1 = tf.cast(e1, tf.float32)
    e0 = tf.cast(e0, tf.float32)
    def loss_fn(the_all, Z, tao=0.1):     # bz=512   tf.shape()
        B = tf.shape(Z)[0]
        Y, entro_word, entro_sum, sort = tf.split(the_all, [1, num, 1, num], axis=-1)
        entro_word1, entro_sum1, sort1 = tf.split(e1, [num, 1, num], axis=-1)
        entro_word0, entro_sum0, sort0 = tf.split(e0, [num, 1, num], axis=-1)

        entro_word = tf.concat([entro_word, entro_word1, entro_word0], axis=0)
        entro_sum = tf.concat([entro_sum, entro_sum1, entro_sum0], axis=0)
        sort = tf.concat([sort, sort1, sort0], axis=0)  # concat (512 24) (1 24) (1 24) ->  (514 24)

        all_prior_sim = compute_iobs(entro_word, entro_sum, sort, num)  # (514 513)
        prior_sim = all_prior_sim[:B,:B-1]   # 512*511
        prior_sim1 = all_prior_sim[:B, B-1]  # 512*1
        prior_sim0 = all_prior_sim[:B, B]    # 512*1

        hard_labels = Y_to_labels(Y)         # 512*511
        logits = Z_to_logits(Z, tao)         # 512*511

        Y = tf.cast(Y, tf.float32)
        NY = tf.cast(1 - Y, tf.float32)
        weight = tf.zeros_like(logits)
        logits = logits + a * prior_sim * (1 - hard_labels) / tao   # obs - negative sample pair
        prior_sim1 = tf.reshape(prior_sim1, (B, 1))
        prior_sim0 = tf.reshape(prior_sim0, (B, 1))
        sim1 = tf.matmul(Z, c1, ) / tao      # BxC x Cx1 = Bx1
        sim0 = tf.matmul(Z, c0, ) / tao
        all_logits = tf.concat([logits, sim1, sim0], axis=-1)
        w1 = - a * prior_sim1 * Y / tao                            # obs - positive sample pair
        w0 = - a * prior_sim0 * NY / tao

        weight = tf.concat([weight, w1, w0], axis=-1)
        all_hard_labels = tf.concat([hard_labels, Y, NY], axis=-1)

        loss = con_loss2(all_logits, all_hard_labels, weight)    # 512*513
        loss = loss / tf.cast(B, tf.float32)
        return loss
    return loss_fn


def Pearson(c1, c0, e1, e0, a ):
    e1 = tf.cast(e1, tf.float32)
    e0 = tf.cast(e0, tf.float32)
    def loss_fn(the_all, Z, tao=0.1):
        B = tf.shape(Z)[0]
        Y, X = tf.split(the_all, [1, 500], axis=-1)
        X = tf.concat([X, e1, e0], axis=0)
        all_prior_sim = compute_pearson(X)
        prior_sim = all_prior_sim[:B,:B-1]  # 128*127
        prior_sim1 = all_prior_sim[:B, B-1]
        prior_sim0 = all_prior_sim[:B, B]

        hard_labels = Y_to_labels(Y)
        logits = Z_to_logits(Z, tao)
        Y = tf.cast(Y, tf.float32)
        NY = tf.cast(1 - Y, tf.float32)
        weight = tf.zeros_like(logits)    # 128*129
        logits = logits + a * prior_sim * (1 - hard_labels) / tao
        prior_sim1 = tf.reshape(prior_sim1, (B, 1))
        prior_sim0 = tf.reshape(prior_sim0, (B, 1))
        sim1 = tf.matmul(Z, c1, ) / tao   # BxC x Cx1 = Bx1
        sim0 = tf.matmul(Z, c0, ) / tao
        all_logits = tf.concat([logits, sim1, sim0], axis=-1)
        w1 = - a * prior_sim1 * Y / tao
        w0 = - a * prior_sim0 * NY / tao

        weight = tf.concat([weight, w1, w0], axis=-1)
        all_hard_labels = tf.concat([hard_labels, Y, NY], axis=-1)

        loss = con_loss2(all_logits, all_hard_labels, weight)
        loss = loss / tf.cast(B, tf.float32)
        return loss

    return loss_fn

