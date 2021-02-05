<template>
  <div id="app">
    <TopBar />
    <router-view />
  </div>
</template>

<script>
import TopBar from '@/components/TopBar.vue';

import '@/styles/base.scss';

export default {
  components: {
    TopBar,
  },
};
</script>
