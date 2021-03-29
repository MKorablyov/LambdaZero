
# m = message

# num_message_pairs * [m_i, m_j, m_k]
# m_ijk = f(m_i, m_j, m_k)                # example f(m_i, m_j, m_k) == m_i @ m_j
# num_message_pairs * m_ijk

# scatter_sum(num_message_pairs * m_ijk, num_message_pairs * 1)