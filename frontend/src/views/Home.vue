<template>
  <div class="home">
    <Welcome />
  </div>
</template>

<script>
import Welcome from '@/components/Welcome.vue';

export default {
  components: {
    Welcome,
  },
};
</script>

<style scoped>
.home {
  margin-top: -80px;
}
</style>
