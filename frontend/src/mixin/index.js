import { mapState, mapMutations } from 'vuex';

export default {
  computed: {
    ...mapState(['cssVars']),
  },
  methods: {
    ...mapMutations(['setState']),
  },
};
