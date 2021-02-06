import IconBase from '@/components/IconBase.vue';
import { mapState, mapMutations } from 'vuex';

export default {
  computed: {
    ...mapState(['cssVars']),
  },
  methods: {
    ...mapMutations(['setState']),
  },
  components: {
    IconBase,
  },
};
